# Results layout

Only lightweight, machine-readable summaries belong here. Full `torch.profiler` Chrome traces and PyTorch memory snapshots are written to ignored `local_artifacts/` directories and must not be committed.

- `benchmark/`: `raw.jsonl` plus the Task 1 `benchmark.csv` summary.
- `profile/`: six-run `run_metadata.json`, a measurement-only `trace_summary.csv`, and a small `runs.jsonl` audit trail. The CSV separates CPU ranges, CPU ops, and physical GPU activities; full Chrome traces stay in ignored `local_artifacts/profile/`.
- `mixed_precision.json`: accumulation, ToyModel dtype, and the paired FP32/BF16 numeric-trend diagnostic. The separate mixed-precision benchmark JSONL/CSV retains timing and peak-memory comparisons; failures are recorded without terminal text.
- `memory/`: `peaks.csv`, `run_metadata.json`, raw run records, and explicit failed-configuration records. CUDA OOM rows retain only structured, path-free allocator telemetry. PyTorch memory snapshots stay in ignored `local_artifacts/memory/`.
- `legacy/`: measurements captured before the guide-compliant profiling implementation; retain for reference only, not as final evidence.

## Inspecting a memory snapshot

To find the largest *single real allocation* without estimating from a memory-timeline rectangle, run the analyzer on a trusted PyTorch memory-history pickle:

```sh
uv run python profiling/analyze_memory_snapshot.py \
  --snapshot local_artifacts/memory/xl_ctx2048_bs1_train_step_fp32.pickle \
  --json-output local_artifacts/memory/xl_ctx2048_bs1_train_step_fp32.analysis.json
```

The terminal report prints the complete stack trace for one representative maximum allocation. The JSON report preserves every distinct stack trace tied at the maximum size, so a report can select the trace with the clearest application-level callsite. The tool calculates the maximum only from `device_traces[*]` events where `action == "alloc"`; it does not infer allocation size from visualizer geometry or allocator segments. PyTorch snapshots are pickle files, so run it only on snapshots from a trusted source.
