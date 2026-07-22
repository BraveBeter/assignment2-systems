# 2.1.3 End-to-End Benchmarking

## 实验设置

- 硬件与软件：NVIDIA H100 80GB HBM3，CUDA 12.8，PyTorch 2.11.0+cu128，Python 3.12.3。
- 模型配置采用 Table 1；`vocab_size=10000`、`context_length=512`、`batch_size=4`。所有数值均为 CUDA Event 计得的毫秒，格式为 `mean ± sample std`，每项有 10 个 measurement step。
- `forward-backward` 和 `full-train` 是累积 workload。因此表中单独的 backward 由 `forward-backward - forward` 估计，optimizer 由 `full-train - forward-backward` 估计；这两列的标准差为两个独立累积测量标准差的传播估计，而非单独 instrument 的 phase-level 标准差。
- 10B 配置在这张 H100 80GB 上发生 OOM，故下表只包含 small、medium、large、xl。

## (b) Five warm-up steps

每个 workload 先执行 5 个不计时 warm-up step，再测量 10 个 step。

| Model | Forward | Forward + backward | Full train | Estimated backward | Estimated optimizer |
|---|---:|---:|---:|---:|---:|
| small | 15.04 ± 1.79 | 40.15 ± 0.11 | 58.26 ± 0.42 | 25.11 ± 1.79 | 18.10 ± 0.43 |
| medium | 29.06 ± 0.20 | 86.09 ± 0.24 | 123.56 ± 1.68 | 57.03 ± 0.31 | 37.47 ± 1.70 |
| large | 46.57 ± 1.35 | 163.78 ± 1.65 | 226.56 ± 0.89 | 117.21 ± 2.13 | 62.79 ± 1.87 |
| xl | 85.14 ± 0.29 | 320.45 ± 0.08 | 514.71 ± 0.85 | 235.30 ± 0.30 | 194.26 ± 0.86 |

**Response.** With five warm-up steps, forward latency grows from 15.04 ms (small) to 85.14 ms (xl), while the estimated backward and optimizer contributions grow from 25.11/18.10 ms to 235.30/194.26 ms, respectively. Variability is generally small relative to the mean (most directly measured workloads have a coefficient of variation below 1.4%); the small and large forward-only measurements are exceptions, with isolated outliers producing 11.9% and 2.9% variation.

## (c) Fewer warm-up steps

此处保持相同的 10 个 measurement step，分别使用 0、1、2 个 warm-up step；任务 (b) 的 5 warm-up 结果是比较基线。

| Model | Warm-up steps | Forward | Forward + backward | Full train |
|---|---:|---:|---:|---:|
| small | 0 | 31.72 ± 52.95 | 69.12 ± 91.80 | 86.95 ± 89.62 |
| small | 1 | 14.96 ± 0.28 | 39.61 ± 0.43 | 58.32 ± 1.86 |
| small | 2 | 14.66 ± 0.21 | 39.69 ± 0.23 | 58.28 ± 1.05 |
| medium | 0 | 49.09 ± 58.79 | 117.47 ± 96.14 | 154.21 ± 92.68 |
| medium | 1 | 29.37 ± 0.98 | 86.22 ± 0.86 | 126.36 ± 6.72 |
| medium | 2 | 29.64 ± 0.29 | 88.29 ± 2.61 | 121.87 ± 0.24 |
| large | 0 | 72.11 ± 82.42 | 199.95 ± 114.56 | 265.29 ± 109.32 |
| large | 1 | 45.82 ± 0.03 | 164.00 ± 0.30 | 231.19 ± 8.76 |
| large | 2 | 45.65 ± 0.03 | 163.70 ± 0.05 | 226.72 ± 0.60 |
| xl | 0 | 111.45 ± 84.66 | 356.78 ± 115.38 | 551.11 ± 119.45 |
| xl | 1 | 85.15 ± 0.24 | 320.35 ± 0.32 | 514.80 ± 0.89 |
| xl | 2 | 84.82 ± 0.24 | 319.53 ± 0.30 | 512.78 ± 0.65 |

**Response.** With no warm-up, the first measured step is dramatically slower and inflates both the mean and standard deviation: for example, small forward has a 182.41 ms first sample followed by roughly 15 ms samples, yielding 31.72 ± 52.95 ms, and xl full-train has an 891.07 ms first sample, yielding 551.11 ± 119.45 ms. One or two warm-up steps bring the means close to the five-warm-up baseline, but some full-train runs still have larger spread (for example, large with one warm-up is 231.19 ± 8.76 ms versus 226.56 ± 0.89 ms with five). The remaining difference is consistent with lazy CUDA-library/kernel initialization, allocator and optimizer-state setup, cache/autotuning effects, and GPU clock state not having fully stabilized after only one or two iterations.
