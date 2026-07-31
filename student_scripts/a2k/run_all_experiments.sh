#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p results

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
  PYTEST_RUNNER=(uv run pytest)
elif command -v python >/dev/null 2>&1; then
  PYTHON_RUNNER=(python)
  PYTEST_RUNNER=(python -m pytest)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_RUNNER=(python3)
  PYTEST_RUNNER=(python3 -m pytest)
else
  printf 'Neither uv nor python/python3 is available.\n' >&2
  exit 127
fi

printf 'Python runner: %s\n' "${PYTHON_RUNNER[*]}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.benchmark_checkpointing
run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.benchmark_attention
run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.benchmark_compile
printf '\n==> %s tests/test_attention.py -v\n' "${PYTEST_RUNNER[*]}"
"${PYTEST_RUNNER[@]}" tests/test_attention.py -v 2>&1 | sed "s|$ROOT|<repo>|g" | tee results/unit_tests.txt
run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.check_flash_attention
run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.benchmark_flash_attention
run "${PYTHON_RUNNER[@]}" -m student_scripts.a2k.plot_flash_results
