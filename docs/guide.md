文档定位。本文围绕 Profiling 部分的作业要求、代码组织、数据采集和 Markdown 报告展开。文档不提供作业答案或预填测量数字；所有结果都应在实际运行环境中独立采集。
实验入口、代码位置、采集目录与提交物
实验仓库：assignment2-systems。本地工作目录可记为 assignment2-systems/
位置
用途
assignment2-systems/profiling/
编写 benchmark、NVTX、memory snapshot 和结果汇总脚本。
assignment2-systems/results/
保存 CSV/JSON、.nsys-rep、torch.profiler trace、snapshot 和图片。
assignment2-systems/writeup.md
最终 Markdown 报告，引用命令、表格、截图和解释。
assignment2-systems/
  profiling/
  results/
    nsys/
    torch/
    memory/
    figures/
  writeup.md
提交关系：代码写在 profiling/，数据抓取到 results/，最终以 writeup.md 为主文件；报告中的每个数字都应能回到对应命令和原始文件。
1. 作业目标、范围与交付物
Profiling 部分的核心是建立一条完整的“测量—定位—解释”链路：先测量一个训练 step 的端到端时间，再把时间分解到 CUDA kernel/算子，最后解释混合精度和显存峰值。重点不是单纯得到一个更快的数字，而是能够说明数字来自什么配置、什么阶段和什么工具。
任务
分值
主要要求
报告中应出现
benchmarking_script
4
支持 forward-only、forward+backward、完整 train-step；包含 warm-up、CUDA 同步和统计。
脚本入口、命令、均值、标准差、warm-up 对照。
nsys_profile
5
选择两个模型规模和三个大于 128 的二次幂 context，分析 forward、backward、optimizer 和 attention。
nsys trace、kernel summary、Calls、NVTX 过滤结果和文字解释。
mixed_precision_accumulation
1
比较四种累加写法，解释低精度累加器和输入量化的影响。
实际输出和 2–3 句误差分析。
benchmarking_mixed_precision
2
完成 ToyModel dtype 分析，比较 FP32 与 BF16 autocast。
dtype 表、时间/显存对照和趋势说明。
memory_profiling
4
分析 XL 在 context 128 和 2048 下的 forward-only 与完整训练步。
memory timeline、峰值表、allocation/stack trace 和 residual/gradient 分析。
实验通常以词表大小 10,000、batch size 4、context length 512 作为统一基线，并使用 small、medium、large、xl、10B 等规模观察变化。长 context 是否能运行必须以实际显存和运行结果为准。
最终交付：主报告为 writeup.md。代码放在 profiling/，原始 CSV/JSON、.nsys-rep、torch trace、snapshot 和图片放在 results/，作为报告的可追溯支撑材料。
2. 代码位置与实验接口
模型本身沿用已经验证过的 Transformer 实现；新增代码主要负责调用模型、控制实验变量、插入标记和保存结果。建议把测量代码集中在 profiling/，不要把计时和 profile 逻辑散落在多个临时 notebook 单元中。
文件
职责
最低要求
profiling/benchmark.py
统一入口和 CLI。
解析配置、生成随机 batch、执行三种 mode、保存 timings。
profiling/nvtx_ranges.py
NVTX 标注。
至少有 measurement、forward、backward、optimizer；attention 可继续细分。
profiling/memory_snapshot.py
显存采集开关。
控制 memory history、snapshot 文件名和实验阶段。
profiling/summarize.py
结果汇总。
把原始数据生成 Markdown 表格，保留单位和统计口径。
writeup.md
最终报告。
记录命令、配置、图表、解释、限制和结论。
2.1 建议的命令行参数
参数
含义
--model-size
模型规模。
--batch-size、--context-length
输入形状。
--mode
forward、forward_backward 或 train_step。
--warmup、--steps
预热步数和正式测量步数。
--dtype
FP32、BF16 或 FP16/autocast 设置。
--seed、--output
复现随机状态和结果文件路径。
参数名称可以不同，但脚本必须能表达这些语义。每次运行都应把完整命令和实际配置写入 metadata。
3. End-to-End Benchmark：怎么做
3.1 三种 mode
mode
包含的操作
注意事项
forward-only
模型 forward，通常使用 no_grad。
只测 logits 产生，不包含 loss、backward 或 optimizer。
forward+backward
forward、loss、backward()。
每步清理梯度，避免跨 step 累积。
train-step
zero_grad、forward、loss、backward、optimizer.step。
用于完整训练步的端到端时间。
3.2 最小执行流程
1. 初始化模型和随机输入，确认一次 tiny forward/backward 能成功。
2. 执行 warm-up；让 CUDA context、算法选择、编译和 allocator 初始化完成。
3. 开始 measurement，用高分辨率计时器包住实际模型执行。
4. 每个被测 CUDA step 后调用 torch.cuda.synchronize()，再记录结束时间。
5. 保存 raw timings、mean、standard deviation、CV、硬件和软件信息。
python profiling/benchmark.py --model-size small --batch-size 4 --context-length 512 --mode train_step --warmup 5 --steps 10 --dtype fp32 --output results/timings.csv
需要单独比较 warm-up 时，保持模型、输入、steps 和计时边界不变，只改变 warm-up 数量。没有同步、把数据生成算进计时、或把首次编译混进 measurement，都会使结果难以解释。
4. Compute Profiling：nsys、torch.profiler 与 Perfetto
必须使用 nsys 的部分：nsys_profile 小题要求读取 CUDA GPU Kernel Summary、kernel Calls、CPU/GPU timeline、CUDA API 与 kernel 的对应关系，以及 NVTX 过滤后的 forward/backward/attention 时间。这些系统级证据不能由一张 torch.profiler operator 表直接替代。
Memory profiling 的例外：显存小问通常用 PyTorch memory history 和 memory_viz；其中 (f) 小问明确要求 Nsight Systems memory trace、按 TransformerBlock 分析 saved residual/gradient，因此必须使用 nsys。
工具
简要用法
最适合回答
Nsight Systems
在 benchmark 前加 nsys profile，生成 .nsys-rep；用 Desktop 打开，或用 nsys stats 导出。
系统级 CUDA kernel、Calls、CPU/GPU timeline、NVTX 和 memory trace。
torch.profiler
短窗口启用 CPU/CUDA activity 和 schedule，导出 operator 表或 Chrome trace JSON。
PyTorch op、autograd、shape、memory 和快速交叉检查。
Perfetto
打开 ui.perfetto.dev，拖入 torch.profiler 导出的 JSON。
查看 Chrome trace 的时间线、slice、搜索和区间细节；不会生成 CUDA kernel 数据。
4.1 nsys 抓取步骤
1. 先用小模型确认 benchmark 和 NVTX 可运行。
2. 只捕获一个稳定的 measurement step，不把 warm-up 纳入统计。
3. 选择两个模型规模和三个 context，分别采集 forward、forward+backward 和 train-step。
4. 打开 .nsys-rep，按 NVTX 过滤，读取 Kernel Summary、Calls 和 CUDA API 对应关系。
5. 在 writeup.md 中记录命令、配置、过滤范围和结论。
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autogradshapes-nvtx --output=results/nsys/<run_name> -- python profiling/benchmark.py --model-size large --context-length 512 --mode train_step --warmup 5 --steps 1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv results/nsys/<run_name>.nsys-rep
4.2 Nsight Systems Desktop UI 怎么看
打开方式有两种：命令行运行 nsys-ui results/nsys/<run_name>.nsys-rep，或启动 Nsight Systems Desktop 后选择 File → Open。CLI 和 Desktop 最好使用兼容版本。
界面区域
看什么
NVTX row
定位 profile/measure、forward、backward 和 attention 子区间。
CUDA API row
查看 CPU 发起的 CUDA 调用；选中一条 API 可以关联到 GPU 上的 kernel。
CUDA HW/GPU row
查看 kernel 的实际执行时长、并行关系和空洞。
Stats System View
打开 CUDA GPU Kernel Summary，记录 kernel 名称、Calls 和累计 GPU 时间。
1. 先在时间线中缩放到一个 measurement step，并用 NVTX 过滤 warm-up。
2. 点击 forward/backward 或 attention 区间，确认对应的 CUDA API 和 GPU kernel。
3. 进入 Stats System View，按累计 GPU 时间排序，再记录 Calls 和过滤条件。
4. 截图时保留模型、context、mode、dtype、时间单位、NVTX 范围和表格标题。
UI 主要用于阅读和截图；无图形界面时，使用 nsys stats 导出同一份报告的统计结果。
[图片]
4.3 NVTX 最小命名规范
至少标记 profile/warmup、profile/measure、forward、backward 和 optimizer。若需要回答 softmax 与 attention matmul 的比较，再增加 attention/scores、attention/softmax、attention/value。
4.4 torch.profiler 与 Perfetto 的关系
torch.profiler 适合用少量 wait/warmup/active steps 检查 operator 和 shape，并设置 record_shapes、profile_memory 或 stack trace。导出 Chrome trace 后，在 Perfetto 中拖入 JSON，使用搜索栏、时间范围和 slice details 查看区间。Perfetto 是 trace 阅读器，不是 CUDA profiler；正式 kernel 归因仍应回到 nsys。
[图片]
5. Mixed Precision 与 Memory Profiling
5.1 Mixed Precision
先完成四种累加实验，观察低精度累加器、FP16 输入量化和 FP32 累加器之间的差异。然后记录 ToyModel 的参数、第一层输出、LayerNorm 输出、logits、loss 和 gradient dtype，并在相同 batch、context、warm-up 和 steps 下比较 FP32 与 BF16 autocast。报告重点是 Tensor Core、reduction、LayerNorm、动态范围、稳定性和显存趋势，不需要预填任何固定数字。
5.2 Memory Profiling
1. 对 XL、context 128 和 2048 分别采集 forward-only 与完整 train-step。
2. warm-up 完成后再开启 memory history，每个配置保存独立 snapshot。
3. 在 memory_viz 中记录 Active Memory Timeline 和峰值，区分 active、allocated、reserved。
4. 推导 residual stream tensor 的理论大小，并与最大 allocation、stack trace 和阶段峰值对照。
5. 按 TransformerBlock 观察 saved residual 释放和 gradient 产生；(f) 小问使用 Nsight memory trace 和 nsys。
显存报告至少应包含：配置表、forward/full-step 峰值、mixed-precision 对照、两张时间线截图，以及对最大 allocation 来源的简短解释。
6. writeup.md 结构与提交要求
# Profiling and Benchmarking

## 1. Environment and code entry
GPU、驱动、CUDA、PyTorch、代码入口

## 2. End-to-end benchmark
配置、warm-up、raw timings、均值和标准差

## 3. Compute profiling
nsys/torch.profiler 命令、NVTX、kernel/Op 归因、截图

## 4. Mixed precision
累加实验、ToyModel dtype、FP32/BF16 对照

## 5. Memory profiling
snapshot、memory_viz、峰值和 residual/gradient 分析

## 6. Limitations and reproducibility
OOM、CUPTI、工具限制和未完成组合
报告内容
最低证据
Benchmark
一次可复现命令、10 次 measurement 的均值/标准差、warm-up 对照。
nsys_profile
.nsys-rep 或导出统计、NVTX 范围、kernel/Calls 说明。
torch.profiler
schedule、activity、trace 路径和它与 nsys 的差异说明。
Memory
snapshot、memory_viz 图、峰值表和 allocation 解释。
主提交文件是 writeup.md。代码放在 profiling/，结果放在 results/。如果平台只接收一个 Markdown 文件，图片使用相对路径或平台附件，命令和关键统计必须写进 Markdown，不能只留在终端。
7. 执行顺序与最终检查
1. 确认模型能完成 tiny forward/backward，固定环境和随机种子。
2. 完成 benchmark，验证同步、warm-up 和三种 mode。
3. 加入 NVTX，先用小模型生成一个可打开的 nsys report。
4. 完成 mixed precision 和 memory snapshot；按要求补采 nsys memory trace。
5. 生成 CSV/JSON、表格和截图，整理到 writeup.md。
6. 逐项检查每个数字是否能追溯到命令、配置和原始文件。
常用资料：Nsight Systems　torch.profiler　memory_viz　Perfetto UI　Perfetto 文档