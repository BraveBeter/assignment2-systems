"""Task 3 accumulation and ToyModel dtype experiments.

Use the unified ``profiling/benchmark.py`` with ``--dtype bf16`` for language
model timing and memory measurements. This module records the two small,
diagnostic experiments required before that benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


DEFAULT_OUTPUT = Path("results/mixed_precision.json")


def accumulation_experiment() -> dict[str, dict[str, float | str]]:
    """Run the four accumulation variants from the handout without rewriting them."""

    variants: tuple[tuple[str, torch.dtype, torch.dtype, bool], ...] = (
        ("fp32_accumulator_fp32_input", torch.float32, torch.float32, False),
        ("fp16_accumulator_fp16_input", torch.float16, torch.float16, False),
        ("fp32_accumulator_fp16_input", torch.float32, torch.float16, False),
        ("fp32_accumulator_explicit_cast_fp16_input", torch.float32, torch.float16, True),
    )
    results: dict[str, dict[str, float | str]] = {}
    for name, accumulator_dtype, input_dtype, explicit_cast in variants:
        total = torch.tensor(0, dtype=accumulator_dtype)
        for _ in range(1000):
            increment = torch.tensor(0.01, dtype=input_dtype)
            if explicit_cast:
                increment = increment.type(torch.float32)
            total += increment
        value = float(total)
        results[name] = {
            "accumulator_dtype": str(accumulator_dtype),
            "input_dtype": str(input_dtype),
            "value": value,
            "absolute_error_from_10": abs(value - 10.0),
        }
    return results


class ToyModel(nn.Module):
    """The exact small model supplied in the mixed-precision handout."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def toy_dtype_experiment(*, autocast_dtype: torch.dtype, device: torch.device) -> dict[str, str]:
    """Measure actual dtypes instead of assuming an autocast policy."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("ToyModel autocast experiment requires CUDA.")
    model = ToyModel(in_features=16, out_features=4).to(device)
    inputs = torch.randn(8, 16, device=device)
    targets = torch.randint(4, (8,), device=device)
    observed: dict[str, str] = {"parameters": str(next(model.parameters()).dtype)}

    def capture(name: str):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = str(output.dtype)

        return hook

    hooks = [model.fc1.register_forward_hook(capture("fc1_output")), model.ln.register_forward_hook(capture("layer_norm_output"))]
    try:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(inputs)
            loss = functional.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()

    observed["logits"] = str(logits.dtype)
    observed["loss"] = str(loss.dtype)
    observed["gradient"] = str(model.fc1.weight.grad.dtype)
    return observed


def load_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_output(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 3 diagnostic mixed-precision experiments.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    for name in ("accumulation", "toy"):
        command = subparsers.add_parser(name)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    toy = subparsers.choices["toy"]
    toy.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    toy.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = load_output(args.output)
    output["timestamp_utc"] = datetime.now(UTC).isoformat()
    if args.experiment == "accumulation":
        output["accumulation"] = accumulation_experiment()
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        output[f"toy_{args.dtype}"] = toy_dtype_experiment(autocast_dtype=dtype, device=torch.device(args.device))
    save_output(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    main()
