# Results layout

Only lightweight, machine-readable summaries belong here. Full `torch.profiler` Chrome traces and PyTorch memory snapshots are written to ignored `local_artifacts/` directories and must not be committed.

- `benchmark/`: `raw.jsonl` plus the Task 1 `benchmark.csv` summary.
- `profile/`: six-run `run_metadata.json`, compact `trace_summary.csv`, and a small `runs.jsonl` audit trail. Full Chrome traces and individual operator CSVs stay in ignored `local_artifacts/profile/`.
- `mixed_precision.json`: accumulation and ToyModel dtype observations. The separate mixed-precision benchmark JSONL/CSV retains timing and peak-memory comparisons; failures are recorded without terminal text.
- `memory/`: `peaks.csv`, `run_metadata.json`, raw run records, and explicit failed-configuration records. PyTorch memory snapshots stay in ignored `local_artifacts/memory/`.
- `legacy/`: measurements captured before the guide-compliant profiling implementation; retain for reference only, not as final evidence.
