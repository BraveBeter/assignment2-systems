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
