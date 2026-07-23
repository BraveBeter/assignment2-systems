from __future__ import annotations

import csv
import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import profiling.benchmark as benchmark_module
from profiling.benchmark import Mode, Precision, RunConfig, build_parser, execute_step, public_command, resolve_configs, validate_auxiliary_args, warmup_partition
import profiling.collect_memory as collect_memory_module
from profiling.collect_memory import failure_record, retry_existing_oom
from profiling.collect_utils import requested_allocation_bytes
from profiling.mixed_precision import accumulation_experiment, numeric_error_metrics, summarize_numeric_steps
from profiling.summarize import read_jsonl, write_benchmark_csv, write_memory_csv
from profiling.trace_summary import TraceSummaryError, metadata_from_records, summarize_trace


def sample_run_config(mode: Mode) -> RunConfig:
    return RunConfig(
        model_size="small",
        mode=mode,
        precision=Precision.FP32,
        warmup_steps=5,
        measurement_steps=10,
        device="cuda",
        seed=0,
        lr_max=1e-3,
        lr_min=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        grad_clip=1.0,
        nvtx=False,
        profile_tool="none",
        track_memory=False,
    )


def sample_record(*, memory: bool = False) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-07-22T00:00:00+00:00",
        "model_config": {"vocab_size": 10_000, "context_length": 512, "batch_size": 4, "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "run_config": {"model_size": "small", "mode": "train_step", "precision": "fp32", "warmup_steps": 5, "measurement_steps": 10},
        "statistics": {"mean_ms": 1.5, "std_ms": 0.1, "cv": 0.0667},
        "raw_timings_ms": [1.4, 1.6],
        "parameter_count": 123,
        "environment": {"device_name": "Test GPU", "torch_version": "2.x", "cuda_version": "12.x"},
        "command": "python profiling/benchmark.py",
        "memory": {
            "snapshot_file": "test.pickle",
            "statistics_bytes": {
                "active_bytes": 10,
                "peak_active_bytes": 20,
                "allocated_bytes": 30,
                "peak_allocated_bytes": 40,
                "reserved_bytes": 50,
                "peak_reserved_bytes": 60,
            },
        }
        if memory
        else None,
    }


def test_parser_resolves_guide_modes_and_model_spec() -> None:
    args = build_parser().parse_args(["--model-size", "small", "--mode", "forward_backward"])
    model, run = resolve_configs(args)

    assert model.d_model == 768
    assert model.context_length == 512
    assert run.mode is Mode.FORWARD_BACKWARD


def test_torch_profiler_requires_both_output_paths() -> None:
    args = build_parser().parse_args(["--profile-tool", "torch"])

    with pytest.raises(ValueError, match="trace-output"):
        validate_auxiliary_args(args)


def test_torch_profiler_captures_only_the_final_warmup_step() -> None:
    torch_profile = replace(sample_run_config(Mode.TRAIN_STEP), profile_tool="torch")

    assert warmup_partition(torch_profile) == (4, 1)
    assert warmup_partition(sample_run_config(Mode.TRAIN_STEP)) == (5, 0)

    no_warmup = replace(torch_profile, warmup_steps=0)
    assert warmup_partition(no_warmup) == (0, 0)


def test_final_warmup_runs_inside_profiler_before_measurement(monkeypatch, tmp_path) -> None:
    events: list[tuple[object, ...]] = []
    profiler_active = False

    class FakeModel:
        def to(self, device: torch.device) -> FakeModel:
            return self

        def get_num_params(self) -> int:
            return 0

    class FakeProfiler:
        def export_chrome_trace(self, path: str) -> None:
            events.append(("export", path))

    class FakeSnapshot:
        def __init__(self, path) -> None:
            self.path = path

        def start(self) -> None:
            events.append(("snapshot_start",))

        def stop_and_dump(self) -> None:
            events.append(("snapshot_stop",))

    class FakeTimer:
        def measure(self, *, stream, execute, device) -> float:
            execute()
            return 1.0

    @contextmanager
    def fake_profiler_context(run):
        nonlocal profiler_active
        profiler_active = True
        try:
            yield FakeProfiler()
        finally:
            profiler_active = False

    @contextmanager
    def fake_phase(name: str, *, nvtx: bool, record_function: bool):
        events.append(("phase_enter", name, profiler_active, record_function))
        try:
            yield
        finally:
            events.append(("phase_exit", name, profiler_active, record_function))

    def fake_execute_step(**kwargs) -> None:
        events.append(("step", kwargs["global_step"], profiler_active))

    monkeypatch.setattr(benchmark_module, "validate_configs", lambda model, run: torch.device("cpu"))
    monkeypatch.setattr(benchmark_module, "build_model", lambda config: FakeModel())
    monkeypatch.setattr(benchmark_module, "random_batch", lambda config, device: (torch.ones(1), torch.ones(1)))
    monkeypatch.setattr(benchmark_module, "execute_step", fake_execute_step)
    monkeypatch.setattr(benchmark_module, "synchronize", lambda device: None)
    monkeypatch.setattr(benchmark_module, "profiler_context", fake_profiler_context)
    monkeypatch.setattr(benchmark_module, "phase", fake_phase)
    monkeypatch.setattr(benchmark_module, "MemorySnapshot", FakeSnapshot)
    monkeypatch.setattr(benchmark_module, "CudaEventTimer", FakeTimer)
    monkeypatch.setattr(benchmark_module, "write_profile_summary", lambda profiler, path: None)
    monkeypatch.setattr(benchmark_module.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(benchmark_module.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(benchmark_module.torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(benchmark_module.torch.cuda, "get_device_name", lambda device: "Test GPU")

    model_config, _ = resolve_configs(build_parser().parse_args([]))
    run = replace(sample_run_config(Mode.FORWARD), profile_tool="torch", measurement_steps=1)
    args = SimpleNamespace(
        memory_snapshot=None,
        trace_output=tmp_path / "trace.json",
        profile_summary=tmp_path / "summary.csv",
    )
    benchmark_module.benchmark(model_config, run, args)

    step_events = [event for event in events if event[0] == "step"]
    assert step_events == [
        ("step", 0, False),
        ("step", 1, False),
        ("step", 2, False),
        ("step", 3, False),
        ("step", 4, True),
        ("step", 5, True),
    ]
    assert ("phase_enter", "profile/warmup", True, True) in events
    assert events.index(("snapshot_start",)) > events.index(("step", 4, True))
    assert events.index(("phase_enter", "profile/measure", True, True)) > events.index(("snapshot_start",))


def test_public_command_removes_absolute_workspace_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    command = public_command(["/usr/bin/python", str(tmp_path / "profiling" / "benchmark.py"), "--output", str(tmp_path / "results" / "raw.jsonl")])

    assert str(tmp_path) not in command
    assert "profiling/benchmark.py" in command
    assert "results/raw.jsonl" in command


def test_forward_mode_disables_gradients() -> None:
    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grad_enabled: bool | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.grad_enabled = torch.is_grad_enabled()
            return x

    model = TrackingModel()
    execute_step(
        model=model,  # type: ignore[arg-type]
        optimizer=None,
        x=torch.ones(1, dtype=torch.long),
        y=torch.ones(1, dtype=torch.long),
        run=sample_run_config(Mode.FORWARD),
        global_step=0,
        nvtx=False,
        record_function=False,
    )

    assert not model.training
    assert model.grad_enabled is False


def test_accumulation_preserves_input_quantization_error() -> None:
    values = accumulation_experiment()

    assert values["fp16_accumulator_fp16_input"]["value"] == 9.953125
    assert values["fp32_accumulator_fp16_input"]["value"] == values["fp32_accumulator_explicit_cast_fp16_input"]["value"]
    assert values["fp16_accumulator_fp16_input"]["absolute_error_from_10"] > values["fp32_accumulator_fp16_input"]["absolute_error_from_10"]


def test_numeric_error_metrics_compute_cpu_fp32_vs_bf16_comparison() -> None:
    fp32_logits = torch.tensor([[0.0, 2.0], [4.0, 0.0]], dtype=torch.float32)
    bf16_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]], dtype=torch.bfloat16)

    metrics = numeric_error_metrics(
        fp32_logits=fp32_logits,
        bf16_logits=bf16_logits,
        fp32_loss=torch.tensor(2.0),
        bf16_loss=torch.tensor(2.5),
    )

    assert metrics["fp32_loss"] == 2.0
    assert metrics["bf16_loss"] == 2.5
    assert metrics["loss_abs_diff"] == 0.5
    assert metrics["loss_relative_diff"] == 0.25
    assert metrics["logits_max_abs_diff"] == 1.0
    assert metrics["logits_rmse"] == pytest.approx(2**-0.5)
    assert metrics["logits_relative_l2_error"] == pytest.approx(10**-0.5)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["all_finite"] is True


def test_numeric_error_metrics_make_nonfinite_values_json_safe() -> None:
    metrics = numeric_error_metrics(
        fp32_logits=torch.tensor([[float("nan"), 1.0]]),
        bf16_logits=torch.tensor([[0.0, 1.0]]),
        fp32_loss=torch.tensor(float("inf")),
        bf16_loss=torch.tensor(1.0),
    )

    assert metrics["fp32_loss"] is None
    assert metrics["loss_abs_diff"] is None
    assert metrics["logits_max_abs_diff"] is None
    assert metrics["top1_agreement"] is None
    assert metrics["fp32_logits_finite"] is False
    assert metrics["fp32_loss_finite"] is False
    assert metrics["all_finite"] is False


def test_numeric_step_summary_reports_range_mean_and_available_values() -> None:
    summary = summarize_numeric_steps(
        [
            {"fp32_loss": 2.0, "bf16_loss": 2.5, "all_finite": True, "fp32_loss_finite": True},
            {"fp32_loss": 4.0, "bf16_loss": None, "all_finite": False, "fp32_loss_finite": True},
        ]
    )

    assert summary["measurement_steps"] == 2
    assert summary["all_steps_finite"] is False
    assert summary["finite_step_counts"]["fp32_loss_finite"] == 2
    assert summary["metrics"]["fp32_loss"] == {"min": 2.0, "mean": 3.0, "max": 4.0, "available_steps": 2}
    assert summary["metrics"]["bf16_loss"] == {"min": 2.5, "mean": 2.5, "max": 2.5, "available_steps": 1}
    assert summary["metrics"]["logits_rmse"] == {"min": None, "mean": None, "max": None, "available_steps": 0}


def test_summaries_keep_raw_timings_and_memory_statistics(tmp_path) -> None:
    record = sample_record(memory=True)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
    records = read_jsonl(raw)
    benchmark_csv = tmp_path / "benchmark.csv"
    memory_csv = tmp_path / "memory.csv"

    write_benchmark_csv(records, benchmark_csv)
    write_memory_csv(records, memory_csv)

    benchmark_row = next(csv.DictReader(benchmark_csv.open(encoding="utf-8")))
    memory_row = next(csv.DictReader(memory_csv.open(encoding="utf-8")))
    assert benchmark_row["raw_timings_ms"] == "[1.4, 1.6]"
    assert benchmark_row["peak_reserved_bytes"] == "60"
    assert memory_row["peak_active_bytes"] == "20"


def test_requested_oom_allocation_parser_keeps_only_safe_numeric_size() -> None:
    assert requested_allocation_bytes("CUDA out of memory. Tried to allocate 1.25 GiB.") == 1_342_177_280
    assert requested_allocation_bytes("RuntimeError: allocate 64 MB at /private/secret") == 64_000_000
    assert requested_allocation_bytes("CUDA OOM without a requested allocation") is None


def test_failure_record_sanitizes_child_oom_telemetry(tmp_path) -> None:
    telemetry_path = tmp_path / "failure.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exception": "cuda_oom",
                "model_config": {"context_length": 2048, "batch_size": 4},
                "run_config": {"model_size": "xl", "mode": "train_step", "precision": "fp32"},
                "failure_scope": "warmup",
                "failure_phase": "forward",
                "peak_scope": "warmup",
                "memory": {
                    "statistics_bytes": {
                        "active_bytes": 1,
                        "peak_active_bytes": 2,
                        "allocated_bytes": 3,
                        "peak_allocated_bytes": 4,
                        "reserved_bytes": 5,
                        "peak_reserved_bytes": 6,
                    },
                    "free_bytes": 7,
                    "total_bytes": 8,
                    "requested_allocation_bytes": 9,
                },
                "environment": {"device_name": "NVIDIA H200", "torch_version": "2.x", "cuda_version": "12.x", "python_version": "3.x"},
                "raw_error": "CUDA out of memory at /private/secret",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(args=["benchmark"], returncode=1, stdout="", stderr="")

    status, record = failure_record(
        model_size="xl",
        context_length=2048,
        batch_size=4,
        mode="train_step",
        dtype="fp32",
        completed=completed,
        failure_output=telemetry_path,
    )

    assert status == "cuda_oom"
    assert record["failure_scope"] == "warmup"
    assert record["failure_phase"] == "forward"
    assert record["peak_scope"] == "warmup"
    assert record["memory"]["statistics_bytes"]["peak_reserved_bytes"] == 6
    assert record["memory"]["free_bytes"] == 7
    assert record["memory"]["requested_allocation_bytes"] == 9
    assert record["oom_telemetry_available"] is True
    assert record["schema_version"] == 2
    assert record["telemetry_schema_version"] == 1
    assert "raw_error" not in record
    assert "/private/secret" not in json.dumps(record)


def test_failure_record_marks_missing_telemetry_as_unavailable_without_stderr() -> None:
    completed = subprocess.CompletedProcess(
        args=["benchmark"],
        returncode=1,
        stdout="",
        stderr="torch.OutOfMemoryError: CUDA out of memory.",
    )
    status, record = failure_record(
        model_size="xl",
        context_length=2048,
        batch_size=4,
        mode="train_step",
        dtype="fp32",
        completed=completed,
    )

    assert status == "cuda_oom"
    assert record["failure_scope"] == "unavailable"
    assert record["peak_scope"] == "unavailable"
    assert record["memory"]["telemetry_status"] == "unavailable"
    assert record["memory"]["statistics_bytes"]["peak_allocated_bytes"] is None
    assert record["oom_telemetry_available"] is False
    assert record["schema_version"] == 2
    assert record["telemetry_schema_version"] is None
    assert "OutOfMemoryError" not in json.dumps(record)


def test_snapshot_dump_error_does_not_mask_active_cuda_oom(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def to(self, device: torch.device) -> FakeModel:
            return self

    class FailingSnapshot:
        def __init__(self, path) -> None:
            self.path = path

        def start(self) -> None:
            pass

        def stop_and_dump(self) -> None:
            raise RuntimeError("snapshot dump failure")

    class FakeTimer:
        def measure(self, *, stream, execute, device) -> float:
            execute()
            return 0.0

    def fake_execute_step(**kwargs) -> None:
        kwargs["failure_context"].phase = "forward"
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(benchmark_module, "validate_configs", lambda model, run: torch.device("cpu"))
    monkeypatch.setattr(benchmark_module, "build_model", lambda config: FakeModel())
    monkeypatch.setattr(benchmark_module, "random_batch", lambda config, device: (torch.ones(1), torch.ones(1)))
    monkeypatch.setattr(benchmark_module, "execute_step", fake_execute_step)
    monkeypatch.setattr(benchmark_module, "MemorySnapshot", FailingSnapshot)
    monkeypatch.setattr(benchmark_module, "CudaEventTimer", FakeTimer)
    monkeypatch.setattr(benchmark_module.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(benchmark_module.torch.cuda, "current_stream", lambda device: object())
    model_config, _ = resolve_configs(build_parser().parse_args([]))
    failure_output = tmp_path / "oom.json"
    args = SimpleNamespace(memory_snapshot=None, failure_output=failure_output)
    run = replace(sample_run_config(Mode.FORWARD), warmup_steps=0, measurement_steps=1)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        benchmark_module.benchmark(model_config, run, args)

    telemetry = json.loads(failure_output.read_text(encoding="utf-8"))
    assert telemetry["failure_scope"] == "measurement"
    assert telemetry["failure_phase"] == "forward"


def test_retry_oom_merges_unexpected_success_without_clearing_existing_results(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "memory"
    output_dir.mkdir()
    original = sample_record(memory=True)
    (output_dir / "runs.jsonl").write_text(json.dumps(original) + "\n", encoding="utf-8")
    old_oom = {
        "model_size": "xl",
        "context_length": 2048,
        "batch_size": 4,
        "mode": "train_step",
        "dtype": "fp32",
        "exception": "cuda_oom",
    }
    (output_dir / "failures.jsonl").write_text(json.dumps(old_oom) + "\n", encoding="utf-8")

    def fake_run_one(**kwargs) -> str:
        assert kwargs["failures"] is None
        record = sample_record(memory=True)
        record["model_config"].update({"context_length": 2048, "batch_size": 4})
        record["run_config"].update({"model_size": "xl", "mode": "train_step", "precision": "fp32"})
        kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output"].write_text(json.dumps(record) + "\n", encoding="utf-8")
        return "success"

    monkeypatch.setattr(collect_memory_module, "require_cuda", lambda: None)
    monkeypatch.setattr(collect_memory_module, "run_one", fake_run_one)
    retry_existing_oom(output_dir=output_dir, snapshots=tmp_path / "snapshots")

    runs = read_jsonl(output_dir / "runs.jsonl")
    assert len(runs) == 2
    assert any(record["model_config"]["context_length"] == 512 for record in runs)
    assert any(record["model_config"]["context_length"] == 2048 for record in runs)
    assert read_jsonl(output_dir / "failures.jsonl") == []
    assert (output_dir / "peaks.csv").is_file()
    assert len(json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))) == 2


def test_retry_oom_replaces_only_target_failure_with_fresh_telemetry(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "memory"
    output_dir.mkdir()
    original = sample_record(memory=True)
    (output_dir / "runs.jsonl").write_text(json.dumps(original) + "\n", encoding="utf-8")
    old_oom = {
        "model_size": "xl",
        "context_length": 2048,
        "batch_size": 4,
        "mode": "train_step",
        "dtype": "bf16",
        "exception": "cuda_oom",
    }
    unrelated_failure = {"model_size": "large", "exception": "subprocess_failed"}
    (output_dir / "failures.jsonl").write_text(
        "\n".join((json.dumps(old_oom), json.dumps(unrelated_failure))) + "\n",
        encoding="utf-8",
    )

    def fake_run_one(**kwargs) -> str:
        kwargs["failure_records"].append(
            {
                **old_oom,
                "schema_version": 2,
                "failure_scope": "warmup",
                "failure_phase": "backward",
                "peak_scope": "warmup",
                "memory": {"telemetry_status": "partial"},
            }
        )
        return "cuda_oom"

    monkeypatch.setattr(collect_memory_module, "require_cuda", lambda: None)
    monkeypatch.setattr(collect_memory_module, "run_one", fake_run_one)
    retry_existing_oom(output_dir=output_dir, snapshots=tmp_path / "snapshots")

    failures = read_jsonl(output_dir / "failures.jsonl")
    assert unrelated_failure in failures
    replacement = next(record for record in failures if record.get("exception") == "cuda_oom")
    assert replacement["failure_scope"] == "warmup"
    assert replacement["failure_phase"] == "backward"
    assert replacement["peak_scope"] == "warmup"
    assert len(read_jsonl(output_dir / "runs.jsonl")) == 1


def test_retry_oom_retains_historical_record_on_non_oom_subprocess_failure(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "memory"
    output_dir.mkdir()
    original = sample_record(memory=True)
    old_oom = {
        "model_size": "xl",
        "context_length": 2048,
        "batch_size": 4,
        "mode": "train_step",
        "dtype": "fp32",
        "exception": "cuda_oom",
    }
    (output_dir / "runs.jsonl").write_text(json.dumps(original) + "\n", encoding="utf-8")
    (output_dir / "failures.jsonl").write_text(json.dumps(old_oom) + "\n", encoding="utf-8")

    monkeypatch.setattr(collect_memory_module, "require_cuda", lambda: None)
    monkeypatch.setattr(collect_memory_module, "run_one", lambda **kwargs: "subprocess_failed")

    with pytest.raises(RuntimeError, match="historical OOM record was retained"):
        retry_existing_oom(output_dir=output_dir, snapshots=tmp_path / "snapshots")

    assert read_jsonl(output_dir / "runs.jsonl") == [original]
    assert read_jsonl(output_dir / "failures.jsonl") == [old_oom]


def trace_event(*, name: str, category: str, timestamp_us: float, duration_us: float, external_id: int | None = None) -> dict[str, object]:
    args: dict[str, object] = {}
    if external_id is not None:
        args["External id"] = external_id
    return {"name": name, "cat": category, "ph": "X", "ts": timestamp_us, "dur": duration_us, "args": args}


def measurement_trace_events() -> list[dict[str, object]]:
    """A small Chrome-trace fixture with one ignored warm-up and one measurement."""

    return [
        trace_event(name="profile/warmup", category="user_annotation", timestamp_us=0, duration_us=90, external_id=1),
        trace_event(name="forward", category="user_annotation", timestamp_us=10, duration_us=20, external_id=2),
        trace_event(name="aten::warmup", category="cpu_op", timestamp_us=12, duration_us=5, external_id=3),
        trace_event(name="warmup_kernel", category="kernel", timestamp_us=15, duration_us=2, external_id=3),
        trace_event(name="profile/measure", category="user_annotation", timestamp_us=100, duration_us=100, external_id=4),
        trace_event(name="forward", category="user_annotation", timestamp_us=110, duration_us=20, external_id=5),
        trace_event(name="attention/scores", category="user_annotation", timestamp_us=115, duration_us=4, external_id=6),
        trace_event(name="backward", category="user_annotation", timestamp_us=135, duration_us=20, external_id=7),
        trace_event(name="optimizer", category="user_annotation", timestamp_us=160, duration_us=20, external_id=8),
        trace_event(name="aten::forward", category="cpu_op", timestamp_us=112, duration_us=6, external_id=9),
        trace_event(name="aten::attention", category="cpu_op", timestamp_us=116, duration_us=2, external_id=10),
        trace_event(name="aten::backward", category="cpu_op", timestamp_us=136, duration_us=7, external_id=11),
        trace_event(name="aten::optimizer", category="cpu_op", timestamp_us=161, duration_us=8, external_id=12),
        trace_event(name="forward_kernel", category="kernel", timestamp_us=113, duration_us=3, external_id=9),
        trace_event(name="attention_kernel", category="kernel", timestamp_us=117, duration_us=1, external_id=10),
        trace_event(name="backward_copy", category="gpu_memcpy", timestamp_us=138, duration_us=4, external_id=11),
        trace_event(name="optimizer_fill", category="gpu_memset", timestamp_us=163, duration_us=5, external_id=12),
    ]


def write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def test_trace_summary_excludes_warmup_and_attributes_physical_cuda_activity(tmp_path) -> None:
    trace_path = tmp_path / "small_ctx256_train_step_fp32.json"
    write_trace(trace_path, measurement_trace_events())

    rows = summarize_trace(trace_path, run_name="small_ctx256_train_step_fp32", require_all_top_phases=True)
    range_rows = [row for row in rows if row["row_type"] == "range"]
    cuda_rows = [row for row in rows if row["row_type"] == "cuda_activity"]
    cpu_rows = [row for row in rows if row["row_type"] == "cpu_op"]

    assert all(row["name"] != "profile/warmup" for row in rows)
    assert next(row for row in range_rows if row["name"] == "profile/measure")["range_duration_us"] == 100.0
    attention = next(row for row in range_rows if row["name"] == "attention/scores")
    assert attention["stage"] == "forward"
    assert attention["inclusive"] == "true"
    assert "do not add" in str(attention["notes"])
    assert {(row["name"], row["stage"], row["activity_type"]) for row in cuda_rows} == {
        ("forward_kernel", "forward", "kernel"),
        ("attention_kernel", "forward", "kernel"),
        ("backward_copy", "backward", "gpu_memcpy"),
        ("optimizer_fill", "optimizer", "gpu_memset"),
    }
    assert sum(int(row["kernel_calls"]) for row in cuda_rows) == 2
    assert all(row["cuda_time_total_us"] == "" for row in range_rows + cpu_rows)
    assert all(row["range_duration_us"] == "" and row["cpu_time_total_us"] == "" for row in cuda_rows)


def test_trace_summary_rejects_unassociated_physical_cuda_activity(tmp_path) -> None:
    trace_path = tmp_path / "invalid.json"
    events = measurement_trace_events()
    events[-1]["args"] = {"External id": 999}
    write_trace(trace_path, events)

    with pytest.raises(TraceSummaryError, match="no unique cpu_op association"):
        summarize_trace(trace_path, run_name="invalid", require_all_top_phases=True)


def test_trace_summary_rejects_missing_or_ambiguous_measurement_ranges(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    missing = [event for event in measurement_trace_events() if event["name"] != "profile/measure"]
    write_trace(missing_path, missing)
    with pytest.raises(TraceSummaryError, match="expected exactly one 'profile/measure'.*found 0"):
        summarize_trace(missing_path, run_name="missing", require_all_top_phases=True)

    ambiguous_path = tmp_path / "ambiguous.json"
    ambiguous = measurement_trace_events()
    ambiguous.append(trace_event(name="profile/measure", category="user_annotation", timestamp_us=300, duration_us=20))
    write_trace(ambiguous_path, ambiguous)
    with pytest.raises(TraceSummaryError, match="expected exactly one 'profile/measure'.*found 2"):
        summarize_trace(ambiguous_path, run_name="ambiguous", require_all_top_phases=True)


def test_trace_summary_keeps_same_named_cpu_and_gpu_activity_in_distinct_rows(tmp_path) -> None:
    trace_path = tmp_path / "same-name.json"
    events = measurement_trace_events()
    for event in events:
        if event["name"] == "forward_kernel":
            event["name"] = "aten::forward"
    write_trace(trace_path, events)

    rows = summarize_trace(trace_path, run_name="same-name", require_all_top_phases=True)
    same_name_rows = [row for row in rows if row["name"] == "aten::forward"]

    assert {(row["row_type"], row["stage"]) for row in same_name_rows} == {
        ("cpu_op", "forward"),
        ("cuda_activity", "forward"),
    }
    assert next(row for row in same_name_rows if row["row_type"] == "cpu_op")["cpu_time_total_us"] == 6.0
    assert next(row for row in same_name_rows if row["row_type"] == "cuda_activity")["cuda_time_total_us"] == 3.0


def test_profile_metadata_is_built_from_audit_environment_without_absolute_paths() -> None:
    record = sample_record()
    record["command"] = "/Users/example/.venv/bin/python /Users/example/project/profiling/benchmark.py --output=/Users/example/project/results/profile/runs.jsonl"
    record["run_config"] = {
        "model_size": "small",
        "mode": "train_step",
        "precision": "fp32",
        "profile_tool": "torch",
        "warmup_steps": 5,
        "measurement_steps": 1,
    }
    record["environment"] = {
        "device_name": "NVIDIA H200",
        "torch_version": "2.11.0+cu128",
        "cuda_version": "12.8",
        "python_version": "3.12.3",
    }

    metadata = metadata_from_records([record], expected_run_names=["small_ctx512_train_step_fp32"])

    assert metadata[0]["environment"]["device_name"] == "NVIDIA H200"
    assert metadata[0]["warmup_protocol"] == {"outside_profiler_steps": 4, "inside_profiler_steps": 1}
    assert "/Users/example" not in metadata[0]["command"]
    assert metadata[0]["trace_file"] == "small_ctx512_train_step_fp32.json"
