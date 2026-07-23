from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import profiling.repair_results as repair_results
from profiling.mixed_precision import summarize_numeric_steps
from profiling.repair_results import (
    MEMORY_SUCCESS_BASELINE,
    OOM_MEMORY_STAT_FIELDS,
    OOM_TARGETS,
    atomic_publish,
    expected_profile_run_names,
    validate_memory_repair,
    validate_numeric_trend,
    validate_profile_outputs,
)


PROFILE_FIELDS = [
    "run_name",
    "row_type",
    "name",
    "stage",
    "activity_type",
    "calls",
    "range_duration_us",
    "cpu_time_total_us",
    "cuda_time_total_us",
    "kernel_calls",
    "inclusive",
    "notes",
]


def _profile_row(**values: object) -> dict[str, object]:
    return {field: "" for field in PROFILE_FIELDS} | values


def _write_complete_profile_fixture(directory: Path) -> tuple[Path, Path]:
    summary = directory / "trace_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PROFILE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for run_name in sorted(expected_profile_run_names()):
            writer.writerow(
                _profile_row(
                    run_name=run_name,
                    row_type="range",
                    name="profile/measure",
                    stage="profile/measure",
                    calls=1,
                    range_duration_us=100,
                    inclusive="true",
                )
            )
            for phase in ("forward", "backward", "optimizer"):
                writer.writerow(
                    _profile_row(
                        run_name=run_name,
                        row_type="range",
                        name=phase,
                        stage=phase,
                        calls=1,
                        range_duration_us=25,
                        inclusive="true",
                    )
                )
            for attention in ("attention/scores", "attention/softmax", "attention/value"):
                writer.writerow(
                    _profile_row(
                        run_name=run_name,
                        row_type="range",
                        name=attention,
                        stage="forward",
                        calls=1,
                        range_duration_us=5,
                        inclusive="true",
                    )
                )
            writer.writerow(
                _profile_row(
                    run_name=run_name,
                    row_type="cpu_op",
                    name="aten::matmul",
                    stage="forward",
                    calls=1,
                    cpu_time_total_us=5,
                )
            )
            writer.writerow(
                _profile_row(
                    run_name=run_name,
                    row_type="cuda_activity",
                    name="matmul_kernel",
                    stage="forward",
                    activity_type="kernel",
                    calls=1,
                    cuda_time_total_us=4,
                    kernel_calls=1,
                )
            )

    metadata = directory / "run_metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "run_name": run_name,
                    "model_size": run_name.split("_", 1)[0],
                    "context_length": int(run_name.split("_ctx", 1)[1].split("_", 1)[0]),
                    "batch_size": 4,
                    "mode": "train_step",
                    "dtype": "fp32",
                    "warmup_steps": 5,
                    "warmup_protocol": {"outside_profiler_steps": 4, "inside_profiler_steps": 1},
                    "measurement_steps": 1,
                    "tool": "torch.profiler",
                    "command": "python profiling/benchmark.py --output results/profile/runs.jsonl",
                    "trace_file": f"{run_name}.json",
                    "summary_file": "trace_summary.csv",
                    "environment": {
                        "device_name": "NVIDIA H200",
                        "torch_version": "2.11.0+cu128",
                        "cuda_version": "12.8",
                        "python_version": "3.12.3",
                    },
                }
                for run_name in sorted(expected_profile_run_names())
            ]
        ),
        encoding="utf-8",
    )
    return summary, metadata


def _success_record(identity: tuple[str, int, int, str, str]) -> dict[str, object]:
    model_size, context_length, batch_size, mode, dtype = identity
    return {
        "timestamp_utc": "2026-07-22T00:00:00+00:00",
        "model_config": {"context_length": context_length, "batch_size": batch_size},
        "run_config": {"model_size": model_size, "mode": mode, "precision": dtype},
        "environment": {"device_name": "NVIDIA H200"},
        "command": "python profiling/benchmark.py --output results/memory/runs.jsonl",
        "memory": {
            "snapshot_file": f"{model_size}_ctx{context_length}_bs{batch_size}_{mode}_{dtype}.pickle",
            "statistics_bytes": {field: 1 for field in OOM_MEMORY_STAT_FIELDS},
        },
    }


def _failure_record(identity: tuple[str, int, int, str, str], *, unavailable: bool = False) -> dict[str, object]:
    model_size, context_length, batch_size, mode, dtype = identity
    values = {field: None if unavailable else 1 for field in OOM_MEMORY_STAT_FIELDS}
    free_and_total = {field: None if unavailable else 1 for field in ("free_bytes", "total_bytes", "requested_allocation_bytes")}
    return {
        "model_size": model_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "mode": mode,
        "dtype": dtype,
        "exception": "cuda_oom",
        "oom_telemetry_available": not unavailable,
        "failure_scope": "warmup",
        "failure_phase": "forward",
        "peak_scope": "warmup",
        "memory": {
            "telemetry_status": "unavailable" if unavailable else "available",
            "unavailable_fields": list(values) + list(free_and_total) if unavailable else [],
            "statistics_bytes": values,
            **free_and_total,
        },
        "environment": {
            "device_name": "NVIDIA H200",
            "torch_version": "2.11.0+cu128",
            "cuda_version": "12.8",
            "python_version": "3.12.3",
        },
    }


def _write_memory_fixture(directory: Path) -> None:
    records = [_success_record(identity) for identity in sorted(MEMORY_SUCCESS_BASELINE)]
    (directory / "runs.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    failures = [_failure_record(identity, unavailable=index == 0) for index, identity in enumerate(sorted(OOM_TARGETS))]
    (directory / "failures.jsonl").write_text("".join(json.dumps(record) + "\n" for record in failures), encoding="utf-8")
    with (directory / "peaks.csv").open("w", encoding="utf-8", newline="") as output_file:
        fields = ["model_size", "mode", "dtype", "batch_size", "context_length", *OOM_MEMORY_STAT_FIELDS]
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model_size": record["run_config"]["model_size"],
                    "mode": record["run_config"]["mode"],
                    "dtype": record["run_config"]["precision"],
                    "batch_size": record["model_config"]["batch_size"],
                    "context_length": record["model_config"]["context_length"],
                    **{field: 1 for field in OOM_MEMORY_STAT_FIELDS},
                }
            )
    (directory / "run_metadata.json").write_text(json.dumps([{"command": "python profiling/benchmark.py"} for _ in records]), encoding="utf-8")


def _numeric_steps() -> list[dict[str, object]]:
    return [
        {
            "measurement_step": index,
            "global_step": index + 4,
            "fp32_loss": 2.0,
            "bf16_loss": 2.1,
            "loss_abs_diff": 0.1,
            "loss_relative_diff": 0.05,
            "logits_max_abs_diff": 0.2,
            "logits_rmse": 0.1,
            "logits_relative_l2_error": 0.01,
            "top1_agreement": 1.0,
            "fp32_loss_finite": True,
            "bf16_loss_finite": True,
            "fp32_logits_finite": True,
            "bf16_logits_finite": True,
            "all_finite": True,
        }
        for index in range(1, 11)
    ]


def test_profile_repair_validation_accepts_complete_measurement_only_artifacts(tmp_path) -> None:
    summary, metadata = _write_complete_profile_fixture(tmp_path)

    validate_profile_outputs(summary, metadata)


def test_memory_repair_validation_accepts_honestly_unavailable_telemetry(tmp_path) -> None:
    _write_memory_fixture(tmp_path)

    validate_memory_repair(tmp_path)


def test_numeric_trend_validation_checks_schema_and_preserves_benchmarks(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "mixed_precision_benchmark.jsonl"
    benchmark.write_text(
        "".join(json.dumps({"model_config": {}, "run_config": {}, "statistics": {}}) + "\n" for _ in range(20)),
        encoding="utf-8",
    )
    monkeypatch.setattr(repair_results, "MIXED_PRECISION_BENCHMARKS", benchmark)
    steps = _numeric_steps()
    payload = {
        "accumulation": {},
        "toy_bf16": {},
        "language_model_numeric_trend": {
            "configuration": {
                "model_size": "small",
                "batch_size": 4,
                "context_length": 512,
                "seed": 0,
                "warmup_steps": 5,
                "measurement_steps": 10,
                "mode": "train_step",
                "fp32_precision": "fp32",
                "bf16_precision": "bf16_autocast",
                "model_config": {"batch_size": 4, "context_length": 512},
            },
            "comparison": {"initialization": "shared_cpu_fp32_state_dict"},
            "environment": {
                "device_name": "NVIDIA H200",
                "torch_version": "2.11.0+cu128",
                "cuda_version": "12.8",
                "python_version": "3.12.3",
            },
            "steps": steps,
            "summary": summarize_numeric_steps(steps),
        },
    }
    path = tmp_path / "mixed_precision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    validate_numeric_trend(path)
    payload["language_model_numeric_trend"]["summary"]["metrics"]["fp32_loss"]["mean"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incorrect fp32_loss mean"):
        validate_numeric_trend(path)


def test_atomic_publish_rolls_back_every_public_file_after_a_mid_group_failure(monkeypatch, tmp_path) -> None:
    destination_one = tmp_path / "public-one.txt"
    destination_two = tmp_path / "public-two.txt"
    staged_one = tmp_path / "staged-one.txt"
    staged_two = tmp_path / "staged-two.txt"
    destination_one.write_text("old one", encoding="utf-8")
    destination_two.write_text("old two", encoding="utf-8")
    staged_one.write_text("new one", encoding="utf-8")
    staged_two.write_text("new two", encoding="utf-8")
    original_replace = Path.replace

    def fail_second_staged_publish(self: Path, target: Path) -> Path:
        if self == staged_two and target == destination_two:
            raise OSError("simulated publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_second_staged_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        atomic_publish(((staged_one, destination_one), (staged_two, destination_two)))

    assert destination_one.read_text(encoding="utf-8") == "old one"
    assert destination_two.read_text(encoding="utf-8") == "old two"
    assert not list(tmp_path.glob(".*.repair-*.bak"))
