# Profiling and Benchmarking

## 1. Environment and code entry

This repository follows [`docs/guide.md`](docs/guide.md). Measurement code is under [`profiling/`](profiling/); compact, submission-safe summaries are under [`results/`](results/); full `torch.profiler` Chrome traces and PyTorch memory snapshots are deliberately kept in ignored `local_artifacts/` directories. The final screenshots required by Tasks 2 and 4 should be cropped, sanitized, compressed, placed in [`assets/`](assets/), and linked from the relevant sections below.

The common entry point is [`profiling/benchmark.py`](profiling/benchmark.py). It creates the model and random CUDA batch outside the measurement interval; runs five warm-up steps by default; then uses one reusable pair of `torch.cuda.Event(enable_timing=True)` events around each workload and calls `torch.cuda.synchronize()` before reading every elapsed time. Each JSONL record contains raw timings in milliseconds, mean, sample standard deviation, CV, resolved configuration, the public GPU/software environment, and the command. No guide-compliant CUDA experiment has been run on this checkout yet, so the CUDA-dependent tables below intentionally contain no invented values.

Run the complete data-collection protocol on the target CUDA host:

```bash
uv run python profiling/collect_benchmark.py
uv run python profiling/collect_profiles.py
uv run python profiling/collect_mixed_precision.py
uv run python profiling/collect_memory.py
```

The pre-guide H100 data is preserved only in [`results/legacy/`](results/legacy/) and is not used below: its forward mode did not use `no_grad`, and its mode names, metadata, and statistics do not meet the current guide.

## 2. End-to-end benchmark

The required FP32 baseline is small, batch size 4, context length 512. The collector invokes `forward`, `forward_backward`, and `train_step` with five warm-up steps and ten measurement steps; it additionally runs `train_step` with zero warm-up. `forward` uses `model.eval()` and `torch.no_grad()`; the two training modes call `optimizer.zero_grad(set_to_none=True)` each step, so gradients cannot accumulate across measurements. Model initialization, optimizer creation, and random-batch construction occur before warm-up and are excluded from timing.

The command below writes raw records to `results/benchmark/raw.jsonl` and the compact table to `results/benchmark.csv`:

```bash
uv run python profiling/collect_benchmark.py
```

After collection, report the mean, sample standard deviation, and CV for all four configurations from `results/benchmark.csv`. Interpret `forward_backward - forward` as the incremental cost of the loss/backward portion and `train_step - forward_backward` as the incremental optimizer/gradient-clipping cost, while noting that these are separately measured end-to-end modes rather than isolated kernels.

The warm-up comparison must use the two `train_step` rows with otherwise identical configuration. The no-warm-up run may include CUDA context setup, allocator growth, lazy library/autotuning work, and first-use optimizer-state allocation. One or two warm-up steps may still differ because not every lazy allocation or algorithm-selection path has necessarily stabilized.

## 3. Compute profiling

The shared profiling tool is `torch.profiler`, with CPU and CUDA activities. [`profiling/collect_profiles.py`](profiling/collect_profiles.py) records six pre-warmed, one-step FP32 `train_step` traces: small and XL at context lengths 256, 512, and 1024. It places `record_function` ranges around `profile/warmup`, `profile/measure`, `forward`, `backward`, `optimizer`, and the attention `scores`, `softmax`, and `value` subphases. This meets the requested `2 x 3` matrix and keeps all configurations on one profiling tool and one timing protocol.

```bash
uv run python profiling/collect_profiles.py
```

The full Chrome traces and per-run operator CSVs are stored only in ignored `local_artifacts/profile/`. The lightweight submission evidence is `results/profile/trace_summary.csv` (operator name, Calls, cumulative CPU/CUDA time, and stage ranges) plus `results/profile/run_metadata.json` (configuration, command, and local trace filename). After collection, inspect a representative trace in Perfetto, add a cropped timeline screenshot here, and compare the `forward`, `backward`, `optimizer`, and attention subranges. The analysis should identify the dominant operators and explain how their Calls and CUDA time vary with context length; it must not claim Nsight-only CUDA API/kernel correlations when using `torch.profiler`.

## 4. Mixed precision

### Accumulation experiment

The four fixed accumulation variants were run directly from [`profiling/mixed_precision.py`](profiling/mixed_precision.py). The recorded values are in [`results/mixed_precision.json`](results/mixed_precision.json).

| Accumulator dtype | Input dtype / conversion | Result after 1,000 additions | Absolute error from 10 |
| --- | --- | ---: | ---: |
| FP32 | FP32 | 10.0001335144 | 0.0001335144 |
| FP16 | FP16 | 9.9531250000 | 0.0468750000 |
| FP32 | FP16 | 10.0021362305 | 0.0021362305 |
| FP32 | FP16 then explicit FP32 cast | 10.0021362305 | 0.0021362305 |

Keeping the accumulator in FP16 produces the largest error because every addition rounds at FP16 precision. The two FP32-accumulator cases agree: explicitly casting does not undo the FP16 quantization of `0.01`, but it avoids further low-precision accumulation error; the small remaining difference from 10 therefore comes from the quantized input.

### ToyModel and language-model comparison

Run the CUDA BF16 dtype capture and matched FP32/BF16 language-model measurements with:

```bash
uv run python profiling/collect_mixed_precision.py
```

The collector records ToyModel parameter, `fc1`, LayerNorm, logits, loss, and gradient dtypes in `results/mixed_precision.json`; it writes ten raw timing samples and active/allocated/reserved peak memory statistics for each available language-model configuration to `results/mixed_precision_benchmark.jsonl` and `.csv`. Any unsupported BF16 configuration or OOM is recorded separately in `results/mixed_precision_failures.jsonl`, rather than being silently omitted.

Once collected, compare FP32 and BF16 for the same model size, mode, batch size, context length, warm-up count, and number of measurements. Keep the ToyModel observation separate from the FP16 accumulation experiment above: autocast can use lower precision for Tensor-Core-friendly compute while retaining selected numerically sensitive reductions in higher precision; BF16's larger exponent range changes the overflow trade-off but does not recover information that was already quantized.

## 5. Memory profiling

[`profiling/collect_memory.py`](profiling/collect_memory.py) runs the required XL matrix: contexts 128 and 2048, `forward` and full `train_step`, FP32 and BF16. Every run completes warm-up before resetting peak counters and before enabling PyTorch memory history. Snapshots are written outside the submission tree in `local_artifacts/memory/`; compact peak statistics are saved in `results/memory/peaks.csv`, while `results/memory/run_metadata.json` records each completed run and `results/memory/failures.jsonl` records failed requested configurations.

```bash
uv run python profiling/collect_memory.py
```

If XL/context-2048 at batch size 4 OOMs, the collector records that exact request and retries XL/context-2048 at batch size 1; only if that also OOMs does it proceed to XL/context-1024 batch size 1 and then Large/context-2048 batch size 1. Thus every successful fallback retains its actual model, context length, and batch size rather than being mislabeled as the original experiment.

For the XL residual stream, one FP32 activation tensor has shape `(batch, context, d_model) = (4, L, 2560)`. Its size is `4 x L x 2560 x 4` bytes: **5 MiB at context 128** and **80 MiB at context 2048**. After opening the post-warm-up snapshots in PyTorch memory_viz, add two cropped Active Memory Timeline images (forward and train step), report active/allocated/reserved and peak statistics without mixing their definitions, and use the largest allocation's stack trace to connect the observed peak to activations, saved residuals, or gradients.

## 6. Limitations and reproducibility

This development host has no available CUDA device, so it cannot produce valid GPU timings, CUDA traces, BF16 autocast observations, or memory snapshots. The collectors now fail before creating result files in that environment; `--dry-run` is available on each collector to inspect commands without CUDA. Run the four commands in Section 1 on the CUDA host, inspect local traces/snapshots there, add the required sanitized screenshots, and then replace the intentionally absent CUDA-dependent discussion with conclusions backed by the generated `results/` files.
