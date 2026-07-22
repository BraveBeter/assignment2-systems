# Benchmark logs

The Section 2.1.3 runners generate the following files in this directory:

- `benchmark_2_1_3b.jsonl`: task (b), using five warm-up steps.
- `benchmark_2_1_3c.jsonl`: task (c), using zero, one, and two warm-up steps.

Each file contains raw successful CUDA timing records, one JSON object per model/workload run. The runners only invoke `cs336_systems.benchmarking_script`; they do not interpret results or edit the writeup.
