"""Collect the six Task 2 torch.profiler traces with a shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from profiling.collect_utils import command_display, failure_kind, require_cuda

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "profile"
LOCAL_TRACES = ROOT / "local_artifacts" / "profile"
LOCAL_SUMMARIES = LOCAL_TRACES / "summaries"
MODEL_SIZES = ("small", "medium")
CONTEXT_LENGTHS = (256, 512, 1024)


def command_for(*, model_size: str, context_length: int, name: str, results: Path, traces: Path, summaries: Path) -> list[str]:
    return [
        sys.executable,
        "profiling/benchmark.py",
        "--model-size",
        model_size,
        "--batch-size",
        "4",
        "--context-length",
        str(context_length),
        "--mode",
        "train_step",
        "--dtype",
        "fp32",
        "--seed",
        "0",
        "--warmup",
        "5",
        "--steps",
        "1",
        "--profile-tool",
        "torch",
        "--trace-output",
        str(traces / f"{name}.json"),
        "--profile-summary",
        str(summaries / f"{name}_ops.csv"),
        "--output",
        str(results / "runs.jsonl"),
    ]


def merge_summaries(*, results: Path, summaries: Path, run_names: list[str]) -> None:
    destination = results / "trace_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer: csv.DictWriter | None = None
        for run_name in run_names:
            source = summaries / f"{run_name}_ops.csv"
            with source.open(encoding="utf-8", newline="") as input_file:
                for row in csv.DictReader(input_file):
                    row["run_name"] = run_name
                    if writer is None:
                        writer = csv.DictWriter(output_file, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)


def append_failure(
    path: Path,
    *,
    model_size: str,
    context_length: int,
    completed: subprocess.CompletedProcess[str],
) -> str:
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_size": model_size,
        "context_length": context_length,
        "batch_size": 4,
        "mode": "train_step",
        "dtype": "fp32",
        "tool": "torch.profiler",
        "stage": "benchmark subprocess",
        "exception": failure_kind(completed),
        "return_code": completed.returncode,
    }
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return failure_kind(completed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the six guide-compliant Task 2 torch.profiler traces.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--trace-dir", type=Path, default=LOCAL_TRACES)
    parser.add_argument("--summary-dir", type=Path, default=LOCAL_SUMMARIES, help="Ignored directory for per-run operator CSVs before aggregation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the six commands without requiring CUDA or writing files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    planned_runs = [
        (model_size, context_length, f"{model_size}_ctx{context_length}_train_step_fp32")
        for model_size in MODEL_SIZES
        for context_length in CONTEXT_LENGTHS
    ]
    run_commands = [
        command_for(model_size=model_size, context_length=context_length, name=run_name, results=args.output_dir, traces=args.trace_dir, summaries=args.summary_dir)
        for model_size, context_length, run_name in planned_runs
    ]
    if args.dry_run:
        for command in run_commands:
            print("Planned:", command_display(command), flush=True)
        return
    try:
        require_cuda()
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs.jsonl").write_text("", encoding="utf-8")
    failures = args.output_dir / "failures.jsonl"
    failures.write_text("", encoding="utf-8")
    run_names: list[str] = []
    metadata: list[dict[str, object]] = []
    for (model_size, context_length, run_name), command in zip(planned_runs, run_commands, strict=True):
        print("Running:", command_display(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            append_failure(failures, model_size=model_size, context_length=context_length, completed=completed)
            continue
        run_names.append(run_name)
        metadata.append(
            {
                "run_name": run_name,
                "model_size": model_size,
                "context_length": context_length,
                "batch_size": 4,
                "mode": "train_step",
                "dtype": "fp32",
                "warmup_steps": 5,
                "measurement_steps": 1,
                "tool": "torch.profiler",
                "command": command_display(command),
                "trace_file": f"{run_name}.json",
                "trace_location": "local_artifacts/profile (not submitted)",
                "summary_file": "trace_summary.csv",
            }
        )
    merge_summaries(results=args.output_dir, summaries=args.summary_dir, run_names=run_names)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
