"""Collect Task 4 peak-memory data and post-warm-up PyTorch snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from profiling.collect_utils import command_display, failure_kind, require_cuda
from profiling.summarize import read_jsonl, write_memory_csv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "memory"
LOCAL_SNAPSHOTS = ROOT / "local_artifacts" / "memory"
MODES = ("forward", "train_step")
DTYPES = ("fp32", "bf16")
RunStatus = Literal["success", "cuda_oom", "subprocess_failed"]


def append_failure(
    path: Path,
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    completed: subprocess.CompletedProcess[str],
) -> RunStatus:
    status = failure_kind(completed)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_size": model_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "mode": mode,
        "dtype": dtype,
        "stage": "benchmark subprocess",
        "exception": status,
        "return_code": completed.returncode,
    }
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return status


def command_for(
    *, model_size: str, context_length: int, batch_size: int, mode: str, dtype: str, output: Path, snapshots: Path
) -> list[str]:
    name = f"{model_size}_ctx{context_length}_bs{batch_size}_{mode}_{dtype}"
    return [
        sys.executable,
        "profiling/benchmark.py",
        "--model-size",
        model_size,
        "--context-length",
        str(context_length),
        "--batch-size",
        str(batch_size),
        "--mode",
        mode,
        "--dtype",
        dtype,
        "--seed",
        "0",
        "--warmup",
        "5",
        "--steps",
        "1",
        "--track-memory",
        "--memory-snapshot",
        str(snapshots / f"{name}.pickle"),
        "--output",
        str(output),
    ]


def run_one(
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    output: Path,
    snapshots: Path,
    failures: Path,
) -> RunStatus:
    command = command_for(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
    )
    print("Running:", command_display(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
    if completed.returncode == 0:
        return "success"
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return append_failure(
        failures,
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
        completed=completed,
    )


def run_with_fallback(*, context_length: int, mode: str, dtype: str, output: Path, snapshots: Path, failures: Path) -> None:
    status = run_one(
        model_size="xl",
        context_length=context_length,
        batch_size=4,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom" or context_length != 2048:
        return

    status = run_one(
        model_size="xl",
        context_length=2048,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom":
        return

    status = run_one(
        model_size="xl",
        context_length=1024,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom":
        return

    run_one(
        model_size="large",
        context_length=2048,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the Task 4 XL memory matrix and explicit OOM fallbacks.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--snapshot-dir", type=Path, default=LOCAL_SNAPSHOTS)
    parser.add_argument("--dry-run", action="store_true", help="Print initial XL commands without requiring CUDA or writing files.")
    return parser


def compact_metadata(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep reproducibility data but omit raw timings and internal filesystem paths."""

    metadata: list[dict[str, object]] = []
    for record in records:
        model = record["model_config"]
        run = record["run_config"]
        memory = record.get("memory") or {}
        metadata.append(
            {
                "timestamp_utc": record["timestamp_utc"],
                "model_size": run["model_size"],
                "mode": run["mode"],
                "dtype": run["precision"],
                "batch_size": model["batch_size"],
                "context_length": model["context_length"],
                "warmup_steps": run["warmup_steps"],
                "measurement_steps": run["measurement_steps"],
                "snapshot_file": memory.get("snapshot_file"),
                "environment": record["environment"],
                "command": record["command"],
            }
        )
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output_dir / "runs.jsonl"
    initial_commands = [
        command_for(
            model_size="xl",
            context_length=context_length,
            batch_size=4,
            mode=mode,
            dtype=dtype,
            output=output,
            snapshots=args.snapshot_dir,
        )
        for dtype in DTYPES
        for context_length in (128, 2048)
        for mode in MODES
    ]
    if args.dry_run:
        for command in initial_commands:
            print("Planned:", command_display(command), flush=True)
        print("If an XL/context-2048 run OOMs, the collector retries batch 1, then XL/context-1024 batch 1, then Large/context-2048 batch 1.")
        return
    try:
        require_cuda()
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    failures = args.output_dir / "failures.jsonl"
    failures.write_text("", encoding="utf-8")
    for dtype in DTYPES:
        for context_length in (128, 2048):
            for mode in MODES:
                run_with_fallback(
                    context_length=context_length,
                    mode=mode,
                    dtype=dtype,
                    output=output,
                    snapshots=args.snapshot_dir,
                    failures=failures,
                )
    records = read_jsonl(output)
    write_memory_csv(records, args.output_dir / "peaks.csv")
    (args.output_dir / "run_metadata.json").write_text(json.dumps(compact_metadata(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
