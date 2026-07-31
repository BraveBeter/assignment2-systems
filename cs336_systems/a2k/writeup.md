# 任务一：Activation Checkpointing

## 1. 理论分析（原题 3.2 / `gradient_checkpointing`）

### (a) 忽略计算代价时的最小峰值 activation memory

把 `N` 个 Transformer block 按近似等长的两半递归划分，并在每一层递归中分别对左右子区间调用 `checkpoint`，直到叶子区间只含一个 block；因此 checkpoint 是平衡、递归且嵌套的，而不是只在最外层切成若干非嵌套段。反向传播进入一个叶子 block 时，活跃内存主要由递归路径上每层的一份边界 activation 与该叶子重算出的 residual 构成，所以峰值 activation memory 为 `O(log N)`（不计参数、梯度、optimizer state，并按单个 block activation 为单位）。每一层递归都会让全部 `N` 个 block 总计再执行常数次，递归深度为 `O(log N)`，故包含重计算的总计算量为 `O(N log N)`。峰值出现在最深叶子区间的反向阶段：此时递归路径上的边界 activation 尚未释放，同时当前 block 的 residual 已被物化。

下面的代码骨架共 9 行，`checkpoint` 调用即为边界：

```python
from torch.utils.checkpoint import checkpoint

def nested_forward(blocks, lo, hi, x):
    if hi - lo == 1:
        return blocks[lo](x)                         # 叶子：物化一个 block 的 residual
    mid = (lo + hi) // 2
    x = checkpoint(lambda z: nested_forward(blocks, lo, mid, z), x,
                   use_reentrant=False)              # 左半区间边界（递归嵌套）
    x = checkpoint(lambda z: nested_forward(blocks, mid, hi, z), x,
                   use_reentrant=False)              # 右半区间边界（递归嵌套）
    return x
```

### (b) 一次重计算、禁止嵌套时的策略

令非嵌套 checkpoint 的 block size 为 `B`。前向结束后需要长期保存约 `ceil(N / B)` 个区间入口 activation，而反向处理某一区间时会短期物化约 `B` 层的 residual，因此忽略固定项后的峰值可写成 `M_peak(B) ≈ (N / B) A + B R`，其中 `A` 是一个边界 activation 的大小、`R` 是一层 residual 的大小；若只看渐近量级且 `A`、`R` 同阶，则最优 `B = Θ(sqrt(N))`，峰值为 `Θ(sqrt(N))`，总计算量仍为 `Θ(N)`（每层额外重算一次）。本任务使用 `N=24`，所以理论平衡点约为 `sqrt(24)≈4.9`；正式实验不预设答案，而是比较无 checkpoint 与 `B∈{1,2,4,8}`，再以实测 peak allocated 最低的成功配置参加 context length 2048 边界实验。

#### RTX 4090 实测结果（待运行）

实验数据将在 RTX 4090 上运行后从 `results/checkpointing.csv` 分析得到；在取得正式数据前不填写或推测测量值。

```bash
uv run python -m student_scripts.a2k.benchmark_checkpointing
```

## 2. 实验设计与可复现性

实验脚本固定使用原题表 1 的 Stanford medium 配置：`d_model=1024`、`d_ff=4096`、24 层、16 heads，并按原题的非 leaderboard 设置使用 vocab size 10000。结合本仓库未绑定 embedding/lm-head 权重且每层采用 SwiGLU 的实现，模型共有 `423,183,360` 个参数（约 423.2M）；FP32 参数本身约占 1.576 GiB，参数、梯度和 AdamW 的两份 FP32 状态合计约 6.306 GiB，尚未包含 activation 与临时 workspace。

实验使用 batch size 1、FP32 参数、BF16 autocast 和 `torch.optim.AdamW`。每个配置都由父进程串行启动一个新的 Python 子进程；子进程在第一次 CUDA tensor/model/optimizer allocation 前设置 23 GiB allocator 上限，并检查当前是单张 RTX 4090 且开始时至少有 22 GiB 空闲显存。每个配置先做 3 个完整 warm-up training steps，再做 5 个 measurement steps；输入、targets 和 seed 在计时区间外固定创建，每次测量都在前后同步 CUDA，并在开始前重置 peak-memory statistics。

对 5 个 latency 原始样本报告 p50；CSV 中的 peak allocated 和 peak reserved 是 5 个独立重置区间中的最大值。标准矩阵完成后，脚本只依据 context 1024 成功行的 peak allocated 选择最佳 checkpoint block size，然后在独立进程中运行 context 2048 的无-checkpoint基线和该最佳配置；OOM 会作为结果行保留，不会被缩小 shape 或静默删除。脚本只负责生成 `results/checkpointing.csv`、`results/run_metadata.json` 和 `results/memory_evidence.json`，报告中的表格与结论在取得正式结果后另行分析撰写。

# 任务二：PyTorch Attention 与 `torch.compile`

## 1. 显式 PyTorch Attention

实验直接复用 `cs336_basics.model.scaled_dot_product_attention`；这是仓库自行实现的显式 PyTorch 函数，按顺序执行 `QK^T`、`1/sqrt(d)` scale、causal mask、softmax 和 `PV`，并非被禁止的 `torch.nn.functional.scaled_dot_product_attention` fused 接口。causal mask 与随机 `Q/K/V` 均在计时区间外创建；输入 shape 为 `[batch, sequence_length, head_dim]`，固定 batch size 1、BF16，并测试 `sequence_length∈{512, 2048, 8192}` 与 `head_dim∈{64, 128}` 的完整笛卡尔积。

baseline 脚本对 forward、backward-only 和 forward-backward 分别使用 `triton.testing.do_bench(warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])`。backward-only 复用一张保留的前向计算图并通过 `torch.autograd.grad(..., retain_graph=True)` 避免把重新前向或梯度累加计入该阶段；forward-backward 每次创建并消费一张新图。每个 shape 在新的 Python 子进程中运行，先设置 23 GiB allocator 上限并核验 RTX 4090 与起始空闲显存；OOM 仍写入 `results/attention_baseline.csv`，不会删除或缩小配置。

```bash
uv run python -m student_scripts.a2k.benchmark_attention
```

## 2. `torch.compile` 对照

attention 对照固定使用 `(512, 64)`、`(2048, 128)`、`(8192, 128)`，eager 与 compiled 使用相同输入、dtype、causal mask、计时阶段和分位数定义。compiled attention 使用 `torch.compile(..., backend="inductor", fullgraph=True, dynamic=False)`；首次 compiled forward 与首次 compiled backward 分开同步计时，随后才测 steady-state latency。每个 compiled 配置使用独立且初始为空的 `TORCHINDUCTOR_CACHE_DIR` 与 `TRITON_CACHE_DIR`，避免跨 shape 或重复运行的磁盘缓存污染 cold-start 数字。

整模型对照使用 Stanford small 配置：vocab size 10000、`d_model=768`、`d_ff=3072`、12 层、12 heads、context length 512、batch size 1、FP32 参数与 BF16 autocast。模型级 compiled 版本使用 `fullgraph=False`，以便把实际 graph break 情况记录到 CSV；forward、backward-only、forward-backward 和包含 AdamW optimizer step 的完整 training step 各做 5 次 warm-up 与 10 次 CUDA-event measurements，并报告 p20/p50/p80。所有对照行和 graph counters 写入 `results/compile_comparison.csv`，环境、编译策略与测量边界写入 `results/run_metadata.json`。

```bash
uv run python -m student_scripts.a2k.benchmark_compile
```

## 3. RTX 4090 实测结果（待填）

### 3.1 显式 Attention Baseline

<!-- 远端运行后，从 results/attention_baseline.csv 填入完整 6 行结果；保留 OOM 行。 -->

（待填）

### 3.2 Eager 与 Compiled Attention

<!-- 远端运行后，从 results/compile_comparison.csv 填入 attention 行，并分开展示 cold-start 与 steady-state。 -->

（待填）

### 3.3 Eager 与 Compiled Stanford Small 模型

<!-- 远端运行后，从 results/compile_comparison.csv 填入 model 行。 -->

（待填）

## 4. 结果分析（待填）

<!-- 取得正式结果后分析以下内容：
1. latency 随 sequence length/head dimension 的变化，以及 forward、backward、forward-backward 的关系；
2. 最早 OOM 配置的显存核算，以及二次方 attention score/softmax 保存量随 sequence length 的变化；
3. compiled 的 cold-start 与 steady-state 收益，不能只写“compiled 更快”；
4. graph break、固定 shape specialization、独立编译缓存与测量稳定性；
5. attention microbenchmark 与整模型/optimizer step 的收益差异。
-->

（待填）

# 任务三：FlashAttention-2 前向

实现位于 `cs336_systems/a2k/attention.py`，并通过 `tests/adapters.py` 暴露
`FlashAttentionPyTorch` 与 `FlashAttentionTriton` 两个 `torch.autograd.Function`。两条路径
接口均为 `apply(Q, K, V, is_causal=False)`，支持 `[batch, sequence, head_dim]` 输入和
causal/non-causal；前向保存 `Q/K/V/O` 及唯一一个 `[batch, n_queries]` 的 FP32
log-sum-exp `L`。

PyTorch reference 逐个 `128×128` tile 维护 FP32 行级 running maximum `m`、normalizer
`l` 与 output accumulator：

```text
m' = max(m, rowmax(S)); P̃ = exp(S - m')
l' = exp(m - m')l + rowsum(P̃)
O' = exp(m - m')O + P̃V; L = m' + log(l')
```

Triton 路径使用自写 `@triton.jit flash_fwd_kernel`：一个 program instance 负责一个
query tile 与 batch index，kernel 内仅循环 key/value tiles；`m/l/accumulator` 均使用 FP32。
当前 launch 配置是 query/key tile `64/64`、`num_warps=4`、`num_stages=2`。causal mask 用
全局 query/key index 比较，masked score 为题面规定的 `-1e6`。

远端复现：

```bash
uv run pytest tests/test_attention.py -v
uv run python -m student_scripts.a2k.check_flash_attention
```

（待填：RTX 4090 型号、commit、官方 tests 的 pass/fail/skip；`results/correctness.json`
会记录三组 seed、head dimension `32/64/128`、两种 mask 下的 `O/L/dQ/dK/dV` 误差。）

# 任务四：FlashAttention-2 重计算反向

纯 PyTorch 路径使用 `flash_attention_backward`；Triton 路径使用三个自写 kernel，均只由保存的
`L` 和输入/输出重算而不保存 attention probability matrix：

```text
D = rowsum(O ⊙ dO); P = exp(QKᵀ / √d - L)
dV = PᵀdO; dS = P ⊙ (dOVᵀ - D)
dQ = dSK / √d; dK = dSᵀQ / √d
```

Triton 先计算 FP32 `D = rowsum(O ⊙ dO)`；随后按 Algorithm 2 分两遍重算 `P`：一个 key-tile
program 独立累加并写回 `dK/dV`，另一个 query-tile program 独立累加并写回 `dQ`。因此不需要
跨 program 同步或 atomic，且瞬时 score/probability 仅为一个 tile。causal 反向复用与前向相同
的 mask，返回梯度顺序为 `Q/K/V/is_causal`。

（待填：远端梯度误差与官方 CUDA tests 结果。）

# 任务五：正确性与性能矩阵（待远端 RTX 4090 运行）

`student_scripts/a2k/benchmark_flash_attention.py` 以 implementation/shape 独立子进程测量
核心矩阵（BF16、batch size 1、causal、sequence length `512/2048/8192`、head dimension
`64/128`）的 eager PyTorch、compiled PyTorch 与 Triton 的 forward、backward、forward-backward。
16384 边界矩阵比较 eager 与 Triton；每行记录 `do_bench(warmup=100, rep=300)` 的
p20/p50/p80、peak allocated/reserved、同 shape eager speedup、status，及 Triton launch
参数；OOM 行会保留。

```bash
uv run python -m student_scripts.a2k.benchmark_flash_attention
```

也可以在远端 GPU 上用一条命令串行运行任务一至任务五；测试输出会脱敏后写入
`results/unit_tests.txt`：

```bash
bash student_scripts/a2k/run_all_experiments.sh
```

矩阵完成后，`student_scripts.a2k.plot_flash_results` 会从成功行生成
`assets/flash_latency.png` 与 `assets/flash_memory.png`。待远端运行后再填入核心/边界矩阵、
OOM/编译失败记录、两张图及分析；`run_metadata.json` 和 `memory_evidence.json` 会保存可复现
环境与显存摘要。
