#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p results

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run uv run python -m student_scripts.a2k.benchmark_checkpointing
run uv run python -m student_scripts.a2k.benchmark_attention
run uv run python -m student_scripts.a2k.benchmark_compile
printf '\n==> uv run pytest tests/test_attention.py -v\n'
uv run pytest tests/test_attention.py -v 2>&1 | sed "s|$ROOT|<repo>|g" | tee results/unit_tests.txt
run uv run python -m student_scripts.a2k.check_flash_attention
run uv run python -m student_scripts.a2k.benchmark_flash_attention
run uv run python -m student_scripts.a2k.plot_flash_results
