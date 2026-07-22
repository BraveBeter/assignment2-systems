
# A2-P：Profiling 与性能分析

本作业要求建立完整的"测量—定位—解释"链路：先正确测量一个训练 step，再把时间归因到 CUDA kernel 和模型阶段，最后解释混合精度与显存峰值。重点不是只给出一个更快的数字，而是让报告中的每个结论都能回到命令、配置和轻量原始数据。

评分标准与核验方式见 [`EVALUATION.md`](EVALUATION.md)。

**文档定位。**本文围绕 Profiling 部分的作业要求、代码组织、数据采集和 Markdown 报告展开。文档不提供作业答案或预填测量数字；所有结果都应在实际运行环境中独立采集。

## 任务总览与分值

`A2-P` 纳入上游的五个小题：

| 上游 problem | 分值 | 本页位置 | 主要要求 |
| --- | --- | --- | --- |
| `benchmarking_script` | 4 | 任务一：End-to-End Benchmark | 支持 forward-only、forward+backward、完整 train-step；包含 warm-up、CUDA 同步和统计。 |
| `nsys_profile` | 5 | 任务二：Compute Profiling | 选择两个模型规模和三个大于 128 的二次幂 context，分析 forward、backward、optimizer 和 attention。 |
| `mixed_precision_accumulation` | 1 | 任务三：四种累加实验 | 比较四种累加写法，解释低精度累加器和输入量化的影响。 |
| `benchmarking_mixed_precision` | 2 | 任务三：FP32 与 BF16 autocast | 完成 ToyModel dtype 分析，比较 FP32 与 BF16 autocast。 |
| `memory_profiling` | 4 | 任务四：Memory Profiling | 分析 XL 在 context 128 和 2048 下的 forward-only 与完整训练步。 |

实验通常以词表大小 10,000、batch size 4、context length 512 作为统一基线，并使用 small、medium、large、xl、10B 等规模观察变化。长 context 是否能运行必须以实际显存和运行结果为准。

## 实验仓库与代码位置

**实验仓库：**[assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)。本地工作目录可记为 `assignment2-systems/`

本地开发目录结构：
```text
assignment2-systems/
├── profiling/
│   ├── benchmark.py        # 统一入口和 CLI
│   ├── nvtx_ranges.py      # NVTX 标注
│   ├── memory_snapshot.py  # 显存采集开关
│   └── summarize.py        # 结果汇总
├── results/                # 本地原始结果，不直接整体提交
│   ├── nsys/
│   ├── torch/
│   ├── memory/
│   └── figures/
└── writeup.md              # 可选工作稿；最终报告写入 SummerQuest 的 README.md
```

| 文件 | 职责 | 最低要求 |
|---|---|---|
| `profiling/benchmark.py` | 统一入口和 CLI | 解析配置、生成随机 batch、执行三种 mode、保存 timings。 |
| `profiling/nvtx_ranges.py` | NVTX 标注 | 至少有 measurement、forward、backward、optimizer；attention 可继续细分。 |
| `profiling/memory_snapshot.py` | 显存采集开关 | 控制 memory history、snapshot 文件名和实验阶段。 |
| `profiling/summarize.py` | 结果汇总 | 把原始数据生成 Markdown 表格，保留单位和统计口径。 |

### 建议的命令行参数

`benchmark.py` 至少要表达：
- `model-size`、`batch-size`、`context-length`；
- `forward`、`forward_backward`、`train_step` 三种 mode；
- `warmup`、`steps`、`dtype`、`seed` 和 `output`。

参数名称可以不同，但脚本必须能表达这些语义。每次运行都要把完整命令、实际配置、硬件/软件版本和结果路径写入 metadata。硬件信息只保留公开且必要的型号与版本，不记录主机名、IP、用户名或内部目录。

## 提交目录结构

提交系统已经完成这部分功能，你无需实现，脚手架会校验固定兄弟仓库，并创建：

```text
students/<同学真名>/assignments/A2-P/
├── README.md                         # 必交：公开 Markdown 主报告
├── submission/
│   └── profiling/
│       └── **/*.py                   # 必交：自己编写的测量与汇总代码
├── results/                          # 必交：轻量、脱敏、机器可读的汇总
│   ├── benchmark.csv
│   ├── profile/
│   │   ├── trace_summary.csv
│   │   └── run_metadata.json
│   ├── mixed_precision.json
│   └── memory/
│       ├── peaks.csv
│       └── run_metadata.json
└── assets/
    └── *.{png,jpg,jpeg,webp,svg}     # 必交至少 3 张关键截图，必须被 README 引用
```

同步脚本只复制 `profiling/**/*.py`，不会复制上游代码、公共测试、`results/`、trace、snapshot 或依赖文件。`results/` 中的轻量汇总和 `assets/` 中的压缩图片由你确认脱敏后放入个人 `A2-P` 目录。

**提交关系：**代码写在 `profiling/`，数据抓取到本地 `results/`，最终以 `A2-P/README.md` 为主报告文件；报告中的每个数字都应能回到对应命令和原始文件。

---

## 任务一：End-to-End Benchmark

实现统一 benchmark 入口，支持三种 mode：

| mode | 必须包含 | 计时边界要求 |
| --- | --- | --- |
| `forward` | forward；通常使用 `no_grad` | 不含 loss、backward 和 optimizer |
| `forward_backward` | forward、loss、`backward()` | 每步清理梯度，不能跨 step 累积 |
| `train_step` | zero grad、forward、loss、backward、optimizer step | 覆盖完整训练 step |

### 最低实验要求

1. 以 small 模型、batch size 4、context length 512、FP32 为统一基线；
2. 三种 mode 各执行至少 5 个 warm-up step 和 10 个 measurement step；
3. 对 `train_step` 额外比较 warm-up 0 与 warm-up 5，其余配置和计时边界保持不变；
4. 每个被测 CUDA step 后调用 `torch.cuda.synchronize()`，数据生成和初始化不得计入；
5. 保存每次 raw timing、均值、样本标准差和变异系数（CV）。

### 最小执行流程

1. 初始化模型和随机输入，确认一次 tiny forward/backward 能成功。
2. 执行 warm-up；让 CUDA context、算法选择、编译和 allocator 初始化完成。
3. 开始 measurement，用高分辨率计时器包住实际模型执行。
4. 每个被测 CUDA step 后调用 `torch.cuda.synchronize()`，再记录结束时间。
5. 保存 raw timings、mean、standard deviation、CV、硬件和软件信息。

```bash
python profiling/benchmark.py --model-size small --batch-size 4 --context-length 512 --mode train_step --warmup 5 --steps 10 --dtype fp32 --output results/benchmark.csv
```
需要单独比较 warm-up 时，保持模型、输入、steps 和计时边界不变，只改变 warm-up 数量。没有同步、把数据生成算进计时、或把首次编译混进 measurement，都会使结果难以解释。
报告必须说明计时器、同步位置、warm-up 边界，以及 warm-up 前后差异的原因。只有均值而没有 raw timing 和标准差，不满足要求。

---

## 任务二：Compute Profiling

### 2.1 六个 `train_step` trace

选择两个模型规模和三个大于 128 的二次幂 context length，形成 `2 × 3 = 6` 个配置。六个 trace 全部使用完整 `train_step`；本小题不要求额外采集 forward-only 或 forward+backward trace。每个配置只捕获一个预热后的稳定 measurement step。

可以使用 **Nsight Systems** 或 **`torch.profiler`** 完成这六个 trace。两种方案同等接受，但六个配置必须使用同一主工具和同一测量口径。

### 2.2 阶段标记规范

标记方式可以是 NVTX 或 `torch.profiler.record_function`，至少包含：
- `profile/warmup`、`profile/measure`；
- `forward`、`backward`、`optimizer`；
- `attention/scores`、`attention/softmax`、`attention/value`。

对六个配置中的每一个，至少保存模型、context、`train_step`、dtype、工具、命令和本地 trace 文件名。提交的 `results/profile/trace_summary.csv` 必须包含主要 op/kernel、Calls、累计 CPU/CUDA 时间和阶段范围。报告还要选一个代表性配置，比较 forward、backward、optimizer 与 attention 子阶段。

### 2.3 方案 A：Nsight Systems

Nsight Systems 可提供 CUDA API 到 GPU kernel 的系统级关联、kernel Calls 和 NVTX 区间。

示例命令：
```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autogradshapes-nvtx \
  --output=results/profile/<run_name> \
  -- python profiling/benchmark.py <你的参数>

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  results/profile/<run_name>.nsys-rep
```

#### Nsight Systems Desktop UI 查看方法

打开方式有两种：命令行运行 `nsys-ui results/profile/<run_name>.nsys-rep`，或启动 Nsight Systems Desktop 后选择 File → Open。CLI 和 Desktop 最好使用兼容版本。

| 界面区域 | 看什么 |
|---|---|
| NVTX row | 定位 `profile/measure`、`forward`、`backward` 和 attention 子区间。 |
| CUDA API row | 查看 CPU 发起的 CUDA 调用；选中一条 API 可以关联到 GPU 上的 kernel。 |
| CUDA HW/GPU row | 查看 kernel 的实际执行时长、并行关系和空洞。 |
| Stats System View | 打开 CUDA GPU Kernel Summary，记录 kernel 名称、Calls 和累计 GPU 时间。 |

操作步骤：
1. 先在时间线中缩放到一个 measurement step，并用 NVTX 过滤 warm-up。
2. 点击 forward/backward 或 attention 区间，确认对应的 CUDA API 和 GPU kernel。
3. 进入 Stats System View，按累计 GPU 时间排序，再记录 Calls 和过滤条件。
4. 截图时保留模型、context、mode、dtype、时间单位、NVTX 范围和表格标题。

UI 主要用于阅读和截图；无图形界面时，使用 `nsys stats` 导出同一份报告的统计结果。

### 2.4 方案 B：`torch.profiler` 与 Perfetto

`torch.profiler` 是可接受的 trace 替代方案。启用 CPU/CUDA activities，用短 schedule 只捕获一个稳定 step，并导出 Chrome trace 到本地。可以在 Perfetto 中查看 op、CUDA kernel、线程、stream 和阶段区间。

`torch.profiler` 适合用少量 wait/warmup/active steps 检查 operator 和 shape，并设置 `record_shapes`、`profile_memory` 或 stack trace。导出 Chrome trace 后，在 Perfetto 中拖入 JSON，使用搜索栏、时间范围和 slice details 查看区间。

> **注意：**Perfetto 是 trace 阅读器，不是 CUDA profiler；`torch.profiler` 不提供与 nsys 完全相同的系统级 CUDA API 关联。选择该方案时，不要伪造 nsys 专属字段。正式 kernel 归因仍应回到 nsys（如果选方案 A）。

### 2.5 提交规范

无论选哪种工具，`.nsys-rep`、SQLite、完整 Chrome trace 和完整 timeline 都不进入 GitHub。只提交六个 run 的轻量汇总、metadata，以及一张能支撑关键分析的裁剪、脱敏、压缩截图。不要提交整张桌面、完整终端或全量 Profile 视图。

---

## 任务三：Mixed Precision

完成两组实验：

### 3.1 累加误差实验

原样运行固定版本 PDF 的 `mixed_precision_accumulation` 小题给出的四段写法，分别讨论低精度累加器、FP16 输入量化和 FP32 累加器的影响；报告实际输出和 2–3 句误差解释。

### 3.2 ToyModel 与 benchmark

ToyModel 必须使用 CUDA BF16 autocast，记录参数、第一层输出、LayerNorm 输出、logits、loss 和 gradient dtype；在相同 batch、context、warm-up 与 steps 下比较 FP32 和 BF16 autocast 的时间、峰值显存和数值趋势。

> **重要区分：**累加误差小题仍保留上游固定的 FP16 输入/累加器对照；这与 ToyModel 和语言模型使用 BF16 autocast 是两组不同的实验，报告中不得混淆。

报告必须区分"输入先被量化造成的误差"和"累加器精度造成的误差"，并说明 reduction、LayerNorm、Tensor Core 与动态范围对结果的影响。不要预填或照抄固定测量数字。

---

## 任务四：Memory Profiling

最低实验矩阵为 XL 模型、context 128 与 2048，分别采集 forward-only 和完整 `train_step`：

1. warm-up 完成后再开启 PyTorch memory history；
2. 每个配置保存独立 snapshot，并用 memory visualizer 读取；
3. 报告 active、allocated、reserved 和峰值，不混用统计口径；
4. 推导 residual stream tensor 的理论大小，并与最大 allocation、stack trace 和阶段峰值对照；
5. 至少提交两张脱敏后的 Active Memory Timeline 截图；
6. 使用 PyTorch memory history，或开启 `profile_memory=True` 的 `torch.profiler`，按 TransformerBlock 解释 saved residual 的释放与 gradient 的产生。

### OOM 降级策略

如果 XL/context 2048 在 batch size 1 仍 OOM，保留失败配置、阶段、异常类型和峰值摘要，再按 XL/context 1024、Large/context 2048 的顺序尝试。**不得静默缩小配置后仍把结果标成 XL/context 2048。**

由平台、CUPTI 或管理员导致的阻塞应在飞书补充文档中记录并联系助教。

### 显存报告内容

显存报告至少应包含：配置表、forward/full-step 峰值、mixed-precision 对照、两张时间线截图，以及对最大 allocation 来源的简短解释。

---

## Markdown 报告要求

最终主报告固定为`writeup.md`。书面分析、命令、表格和图表统一使用 Markdown；不提交 PDF、Office 文档、notebook 或 notebook 导出文件。

### 报告结构 (writeup.md)

```markdown
# Profiling and Benchmarking

## 1. Environment and code entry
完成范围、未完成项、题面版本、固定 starter commit；
GPU、驱动、CUDA、PyTorch、工具版本（公开脱敏）

## 2. End-to-end benchmark
benchmark 命令、配置、raw timings、均值、标准差、CV、warm-up 对照

## 3. Compute profiling
六个 train_step Profile 配置、所选工具、阶段标记、
op/kernel Calls、CPU/CUDA 时间汇总、代表性 timeline 与归因解释、截图

## 4. Mixed precision
累加实验实际输出、ToyModel dtype 表、FP32/BF16 时间/显存对照、误差分析

## 5. Memory profiling
snapshot、memory_viz 图、峰值表、最大 allocation、residual/gradient 分析、至少两张时间线

## 6. Limitations and reproducibility
OOM、CUPTI、工具或资源限制、仍可复现的最小命令、飞书补充文档链接
```

### 最低证据要求

| 报告内容 | 最低证据 |
|---|---|
| Benchmark | 一次可复现命令、10 次 measurement 的均值/标准差、warm-up 对照。 |
| Profile | 轻量汇总 CSV、metadata、NVTX 范围、kernel/Calls 说明、关键截图。 |
| Mixed Precision | 累加实际输出、dtype 表、时间/显存对照和误差分析。 |
| Memory | 峰值表、snapshot metadata、至少两张时间线截图、allocation 解释。 |

报告中的每个数字都必须能回到 `results/` 的一行数据、一个 metadata 文件或一条明确命令。图片必须使用相对路径并包含有意义的 alt text，不能只粘贴无法搜索或无法解释的截图。

---

## 文件与附件限制

为保证公开仓库可审查、可 clone，`A2-P` 使用比 GitHub 平台上限更严格的规则：

| 范围 | 限制 |
| --- | ---: |
| 目录内任意单文件 | 不超过 5 MiB（仓库统一硬限制） |
| `A2-P` 的 `README.md` | 不超过 1 MiB |
| `results/` 与 `assets/` 公开附件合计 | 不超过 2 MiB |

### 允许提交
- `submission/profiling/**/*.py`；
- `results/**/*.{csv,json,jsonl,md,txt}`；
- `assets/**/*.{png,jpg,jpeg,webp,svg}`。

### 明确禁止提交
- `.nsys-rep`、SQLite、memory snapshot、pickle、完整 Chrome trace；
- 压缩包、数据集、模型权重、checkpoint、虚拟环境、缓存和依赖锁；
- PDF、Office 文档、notebook 与 notebook 导出；
- 未裁剪的终端截图、主机名、IP、用户名、内部路径、UUID、进程列表和任何凭据。

> 附件指 `results/` 与 `assets/` 中的轻量汇总、metadata 和图片；`README.md` 与 `submission/` 代码不计入 2 MiB 附件限额。截图应先裁剪到关键时间段或表格，再用 PNG 压缩、WebP 或其他无损/高质量方式缩小。Profile 只提交支撑结论的关键部分。

大型 profiler 原始文件默认留在个人工作目录，不作为提交物。助教抽查时再按指定的组内受控方式提供；不要为了"证明做过"把大型原始文件上传到公开 GitHub 或随意附在飞书正文。

---

## 执行顺序与最终验收清单

### 建议执行顺序

1. 确认模型能完成 tiny forward/backward，固定环境和随机种子。
2. 完成 benchmark，验证同步、warm-up 和三种 mode。
3. 加入 NVTX / record_function，先用小模型生成一个可打开的 trace。
4. 完成 mixed precision 和 memory snapshot。
5. 生成 CSV/JSON、表格和截图，整理到 `A2-P/README.md`。
6. 逐项检查每个数字是否能追溯到命令、配置和原始文件。

### 最终验收清单

- [ ] 三种 benchmark mode、同步、warm-up 和统计口径完整。
- [ ] 已用 nsys 或 `torch.profiler` 完成两个模型规模、三个 context 的六个 `train_step` trace。
- [ ] Profile 只提交轻量汇总和关键截图，未上传完整 trace。
- [ ] mixed precision 同时覆盖累加误差、dtype、时间和显存。
- [ ] memory profiling 覆盖规定矩阵或如实记录 fallback。
- [ ] `README.md` 是完整 Markdown 主报告，所有数字都可追溯。
- [ ] 代码、汇总和图片位于固定目录，文件类型与大小通过校验。
- [ ] 未提交 trace、snapshot、权重、数据、压缩包、内部信息或凭据。

---

## 常用资料

[Nsight Systems](https://docs.nvidia.com/nsight-systems/)　[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)　[PyTorch Memory Visualizer](https://pytorch.org/memory_viz)　[Perfetto UI](https://ui.perfetto.dev/)　[Perfetto 文档](https://perfetto.dev/docs/)