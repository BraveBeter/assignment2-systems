from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import profiling.benchmark as benchmark_module
from profiling.benchmark import Mode, Precision, RunConfig, build_parser, execute_step, public_command, resolve_configs, validate_auxiliary_args, warmup_partition
from profiling.mixed_precision import accumulation_experiment
from profiling.summarize import read_jsonl, write_benchmark_csv, write_memory_csv


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
