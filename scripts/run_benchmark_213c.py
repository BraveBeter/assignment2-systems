#!/usr/bin/env python3
"""Run the Section 2.1.3(c) warm-up comparison and append raw results to JSONL."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "benchmark_2_1_3c.jsonl"
MODEL_SIZES = ("small", "medium", "large", "xl", "10B")
MODES = ("forward", "forward-backward", "full-train")
WARMUP_STEPS = (0, 1, 2)


def main() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    for warmup_steps in WARMUP_STEPS:
        for model_size in MODEL_SIZES:
            for mode in MODES:
                command = [
                    sys.executable,
                    "-m",
                    "cs336_systems.benchmarking_script",
                    "--model-size",
                    model_size,
                    "--mode",
                    mode,
                    "--warmup-steps",
                    str(warmup_steps),
                    "--steps",
                    "10",
                    "--output",
                    str(LOG_PATH),
                ]
                print("Running:", " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
