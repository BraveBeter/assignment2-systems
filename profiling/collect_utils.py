"""Shared, non-measuring helpers for the CUDA experiment collectors."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch


def require_cuda() -> None:
    """Fail before touching result files when this host cannot run the experiment."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this collector on the target CUDA machine; no result files were changed."
        )


def command_display(command: Sequence[str]) -> str:
    """Return a readable, reproducible command without an absolute interpreter path."""

    executable = "python" if command and command[0].endswith(("python", "python3")) else command[0]
    root = Path.cwd().resolve()

    def visible(argument: str) -> str:
        path = Path(argument)
        if not path.is_absolute():
            return argument
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return path.name

    return " ".join(visible(argument) for argument in (executable, *command[1:]))


def failure_kind(completed: subprocess.CompletedProcess[str]) -> str:
    """Classify an expected CUDA OOM without persisting terminal output or host paths."""

    text = f"{completed.stdout}\n{completed.stderr}".lower()
    return "cuda_oom" if "outofmemoryerror" in text or "out of memory" in text else "subprocess_failed"
