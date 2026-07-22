"""CUDA end-to-end benchmarking for the CS336 basics Transformer.

The script measures one workload at a time.  It intentionally creates the model,
optimizer, and random batch before the warm-up phase so that the reported samples
contain only the requested training-step work.

Example:
    uv run python -m cs336_systems.benchmarking_script \
        --model-size small --mode full-train --warmup-steps 5 --steps 10
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr


@dataclass(frozen=True)
class ModelSpec:
    """The model-size columns specified in Assignment 2, Table 1."""

    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelSpec(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


class Workload(StrEnum):
    """The mutually exclusive execution paths required by Section 2.1.3."""

    FORWARD = "forward"
    FORWARD_BACKWARD = "forward-backward"
    FULL_TRAIN = "full-train"


@dataclass(frozen=True)
class ModelConfig:
    """Fully resolved Transformer construction parameters."""

    vocab_size: int
    context_length: int
    batch_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration that affects execution or measurement of one benchmark."""

    model_size: str
    workload: Workload
    warmup_steps: int
    measurement_steps: int
    device: str
    seed: int
    lr_max: float
    lr_min: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip: float
    nvtx: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Raw samples and derived statistics for a completed benchmark."""

    model_size: str
    model_config: ModelConfig
    benchmark_config: BenchmarkConfig
    parameter_count: int
    sample_times_ms: list[float]
    mean_ms: float
    std_ms: float
    timestamp_utc: str
    device_name: str

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable, self-contained experiment record."""

        return {
            "timestamp_utc": self.timestamp_utc,
            "unit": "milliseconds",
            "model_size": self.model_size,
            "parameter_count": self.parameter_count,
            "model_config": asdict(self.model_config),
            "benchmark_config": {
                **asdict(self.benchmark_config),
                "workload": self.benchmark_config.workload.value,
            },
            "environment": {
                "device_name": self.device_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "python_version": platform.python_version(),
            },
            "sample_times_ms": self.sample_times_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing process arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark CUDA Transformer forward, backward, and full training steps."
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), default="small", help="Base configuration from Assignment 2, Table 1.")
    parser.add_argument("--mode", choices=[workload.value for workload in Workload], default=Workload.FULL_TRAIN.value, help="Workload to measure.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Steps run before timing begins.")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed measurement steps.")
    parser.add_argument("--device", default="cuda", help="CUDA device, for example cuda or cuda:0.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for model weights and random batch generation.")

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument("--vocab-size", type=int, default=10_000)
    model_group.add_argument("--context-length", type=int, default=512)
    model_group.add_argument("--batch-size", type=int, default=4)
    model_group.add_argument("--d-model", type=int, default=None)
    model_group.add_argument("--d-ff", type=int, default=None)
    model_group.add_argument("--num-layers", type=int, default=None)
    model_group.add_argument("--num-heads", type=int, default=None)

    optimizer_group = parser.add_argument_group("optimizer options for full-train")
    optimizer_group.add_argument("--lr-max", type=float, default=1e-3)
    optimizer_group.add_argument("--lr-min", type=float, default=1e-4)
    optimizer_group.add_argument("--weight-decay", type=float, default=0.01)
    optimizer_group.add_argument("--beta1", type=float, default=0.9)
    optimizer_group.add_argument("--beta2", type=float, default=0.999)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL file to append with the completed result.")
    parser.add_argument("--nvtx", action="store_true", help="Wrap the measurement phase in an NVTX range named benchmark_measurement.")
    return parser


def resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    """Merge a Table 1 base spec with explicit command-line overrides."""

    spec = MODEL_SPECS[args.model_size]
    return ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        d_model=spec.d_model if args.d_model is None else args.d_model,
        d_ff=spec.d_ff if args.d_ff is None else args.d_ff,
        num_layers=spec.num_layers if args.num_layers is None else args.num_layers,
        num_heads=spec.num_heads if args.num_heads is None else args.num_heads,
    )


def resolve_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert parsed CLI arguments into a typed benchmark configuration."""

    return BenchmarkConfig(
        model_size=args.model_size,
        workload=Workload(args.mode),
        warmup_steps=args.warmup_steps,
        measurement_steps=args.steps,
        device=args.device,
        seed=args.seed,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        nvtx=args.nvtx,
    )


def validate_config(model_config: ModelConfig, benchmark_config: BenchmarkConfig) -> torch.device:
    """Validate user-facing inputs before any model allocation or output write."""

    positive_values = {
        "vocab_size": model_config.vocab_size,
        "context_length": model_config.context_length,
        "batch_size": model_config.batch_size,
        "d_model": model_config.d_model,
        "d_ff": model_config.d_ff,
        "num_layers": model_config.num_layers,
        "num_heads": model_config.num_heads,
        "measurement_steps": benchmark_config.measurement_steps,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}.")

    if benchmark_config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {benchmark_config.warmup_steps}.")
    if model_config.d_model % model_config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if not 0.0 <= benchmark_config.lr_min <= benchmark_config.lr_max:
        raise ValueError("Require 0 <= lr_min <= lr_max.")
    if not 0.0 <= benchmark_config.weight_decay:
        raise ValueError("weight_decay must be non-negative.")
    if not 0.0 <= benchmark_config.beta1 < 1.0 or not 0.0 <= benchmark_config.beta2 < 1.0:
        raise ValueError("beta1 and beta2 must be in [0, 1).")
    if benchmark_config.eps < 0.0:
        raise ValueError("eps must be non-negative.")
    if benchmark_config.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive.")

    try:
        device = torch.device(benchmark_config.device)
    except RuntimeError as error:
        raise ValueError(f"Invalid device {benchmark_config.device!r}: {error}") from error
    if device.type != "cuda":
        raise ValueError("This benchmark supports CUDA devices only; pass --device cuda or --device cuda:N.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Run this script in a CUDA-enabled PyTorch environment.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def build_model(model_config: ModelConfig) -> BasicsTransformerLM:
    """Construct the baseline Transformer from a resolved model configuration."""

    return BasicsTransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
    )


def generate_random_batch(model_config: ModelConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one random input/target batch before warm-up and reuse it for every step."""

    batch_shape = (model_config.batch_size, model_config.context_length)
    x = torch.randint(model_config.vocab_size, batch_shape, device=device, dtype=torch.long)
    y = torch.randint(model_config.vocab_size, batch_shape, device=device, dtype=torch.long)
    return x, y


def run_workload(
    *,
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    workload: Workload,
    global_step: int,
    benchmark_config: BenchmarkConfig,
) -> None:
    """Execute exactly one requested workload, without timing or synchronization."""

    if workload is not Workload.FORWARD:
        optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if workload is Workload.FORWARD:
        return

    loss = cross_entropy(logits, y)
    loss.backward()
    if workload is Workload.FORWARD_BACKWARD:
        return

    clip_gradient(model.parameters(), benchmark_config.grad_clip)
    learning_rate = get_cosine_lr(
        global_step + 1,
        benchmark_config.lr_max,
        benchmark_config.lr_min,
        benchmark_config.warmup_steps,
        benchmark_config.warmup_steps + benchmark_config.measurement_steps,
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    optimizer.step()


def _synchronize(device: torch.device) -> None:
    """Synchronize the requested CUDA device with the host."""

    torch.cuda.synchronize(device)


def benchmark(model_config: ModelConfig, benchmark_config: BenchmarkConfig) -> BenchmarkResult:
    """Run warm-up and measurement phases and return raw timing samples and statistics."""

    device = validate_config(model_config, benchmark_config)
    torch.manual_seed(benchmark_config.seed)
    torch.cuda.manual_seed_all(benchmark_config.seed)

    model = build_model(model_config).to(device)
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=benchmark_config.lr_max,
        betas=(benchmark_config.beta1, benchmark_config.beta2),
        eps=benchmark_config.eps,
        weight_decay=benchmark_config.weight_decay,
    )
    x, y = generate_random_batch(model_config, device)

    for global_step in range(benchmark_config.warmup_steps):
        run_workload(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            workload=benchmark_config.workload,
            global_step=global_step,
            benchmark_config=benchmark_config,
        )
        _synchronize(device)

    sample_times_ms: list[float] = []
    stream = torch.cuda.current_stream(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if benchmark_config.nvtx:
        torch.cuda.nvtx.range_push("benchmark_measurement")
    try:
        for measurement_index in range(benchmark_config.measurement_steps):
            global_step = benchmark_config.warmup_steps + measurement_index
            start_event.record(stream)
            run_workload(
                model=model,
                optimizer=optimizer,
                x=x,
                y=y,
                workload=benchmark_config.workload,
                global_step=global_step,
                benchmark_config=benchmark_config,
            )
            end_event.record(stream)
            _synchronize(device)
            sample_times_ms.append(start_event.elapsed_time(end_event))
    finally:
        if benchmark_config.nvtx:
            torch.cuda.nvtx.range_pop()

    mean_ms = statistics.mean(sample_times_ms)
    std_ms = statistics.stdev(sample_times_ms) if len(sample_times_ms) > 1 else 0.0
    return BenchmarkResult(
        model_size=benchmark_config.model_size,
        model_config=model_config,
        benchmark_config=benchmark_config,
        parameter_count=model.get_num_params(),
        sample_times_ms=sample_times_ms,
        mean_ms=mean_ms,
        std_ms=std_ms,
        timestamp_utc=datetime.now(UTC).isoformat(),
        device_name=torch.cuda.get_device_name(device),
    )


def append_jsonl(path: Path, result: BenchmarkResult) -> None:
    """Append one completed benchmark record to a JSON Lines output file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(result.to_record(), sort_keys=True))
        output_file.write("\n")


def main(argv: Sequence[str] | None = None) -> BenchmarkResult:
    """Parse arguments, execute one benchmark, print its record, and optionally persist it."""

    parser = build_parser()
    args = parser.parse_args(argv)
    model_config = resolve_model_config(args)
    benchmark_config = resolve_benchmark_config(args)

    try:
        result = benchmark(model_config, benchmark_config)
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(result.to_record(), indent=2, sort_keys=True))
    if args.output is not None:
        append_jsonl(args.output, result)
    return result


if __name__ == "__main__":
    main()
