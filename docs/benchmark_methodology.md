# Benchmark Methodology (Person 3)

This document states exactly what was measured, how, and what the numbers can and cannot support.
Everything here is implemented in `benchmarks/` and can be reproduced from `notebooks/thermal_hpc_case_study.ipynb`.

## 1. Controlled conditions

| Item | Value (identical for every implementation) |
|---|---|
| Model | explicit 2-D finite difference, `T' = T + r(up+down+left+right-4T)` |
| r | 0.2 (stability limit 0.25) |
| Precision | float32 |
| Initial grid | 25 degC plate, 50x50 cell 100 degC hotspot at the centre |
| Boundary | outer cells fixed at 25 degC, never written |
| Buffers | two arrays, pointer swap each step (no in-place update) |
| Compiler flags | CPU `gcc -O3 -ffp-contract=off -fopenmp`; GPU `nvcc -O3 --fmad=false` (no fused multiply-add on either side, so arithmetic is identical) |

## 2. What each timer covers

| Name | CPU | GPU |
|---|---|---|
| **Kernel-only** | wall clock around the K-step loop | `cudaEvent` pair around the K kernel launches (device buffers already resident) |
| **End-to-end** | same as kernel-only (data is already in host RAM) | 2 host-to-device copies + K kernels + 1 device-to-host copy |
| Excluded from both | process start-up, buffer allocation, initial-grid generation, CSV output, CUDA context creation, compilation | same |

Consequences for how results may be quoted:

* A CPU-vs-GPU speedup must say which GPU time it uses. **Kernel-only speedup** answers "how much faster is the computation"; **end-to-end speedup** answers "how much faster is the whole run when data starts and ends on the host".
* Serial-CPU and OpenMP-CPU speedups are separate comparisons. The headline CPU-vs-GPU number should use the OpenMP baseline, because the serial baseline uses one core only.
* The `cpu_bench` serial loop is the fair single-thread baseline. Person 1's original `cpu_serial.c` re-fills the entire next grid every step (an extra full-grid memory pass the other implementations do not perform); `cpu_bench.c` removes it. Its output is bit-identical to Person 1's CSV (`tests/test_correctness.py::test_matches_person1_reference_csv`).

## 3. Measurement protocol

1. **Warm-up** inside every program, never timed: 10 steps (CUDA: context creation, clock ramp-up, caches; CPU: page faults, OpenMP thread pool).
2. **Repetitions:** each CUDA process times 7 repetitions; each CPU process times 1. Every configuration is run in 3 **rounds**.
3. **Interleaving:** within a round the job order is shuffled (fixed seed), so slow drift (GPU clocks, noisy neighbours on a shared VM) is spread across implementations rather than biasing one.
4. **Summary statistic:** median over all repetitions and rounds, with the inter-quartile range. Min/max and the round-to-round range are also stored. Configurations with IQR or round range above 8 % are listed in Table 5 of `summary_tables.md` and must be reported with their spread.
5. **Resolution:** the CUDA harness prints times to 1 microsecond. Runs shorter than about 1 ms (very small N with small K) therefore carry a visible quantisation error; do not over-interpret them.
6. **Environment capture:** `nvidia-smi`, `nvcc --version`, `lscpu`, compiler versions, git commit and flags are saved to `results/env/` automatically.

## 4. Experiments

| ID | Question | Varied | Fixed |
|---|---|---|---|
| Scaling | Where does the GPU overtake the CPU, and how does the gap grow? | N = 128 ... 4096 (8192 optional) | K = 1000 |
| Blocks | Does block geometry matter? Does tiling help? | 8x8, 16x16, 32x8, 32x16, 32x32 x {naive, tiled} | K = 1000, N = 1024 and 4096 |
| Steps | When do launch overhead and PCIe copies matter? | K = 10 ... 10000 | N = 1024, naive 32x8 |
| Threads | How well does OpenMP scale on the VM's CPU? | 1, 2, 4, all threads | N = 2048, K = 1000 |

## 5. Derived metrics

* Updates `U = (N-2)^2 * K`; throughput `U / T_kernel` in Gcell-updates/s.
* Speedup `T_cpu / T_gpu`; runtime reduction `(T_cpu - T_gpu)/T_cpu`; optimisation speedup `T_naive / T_tiled` at identical (N, K, block).
* **Bandwidth.** The team plan used 24 B per update (5 reads + 1 write, no reuse). That is an upper-bound traffic count, not what DRAM sees: on the T4 it produces values above 100 % of the physical peak, which proves the caches absorb most of those reads. The main figure here is therefore **8 B per update** (one read and one write of a float32, assuming perfect neighbour reuse), reported as a percentage of the theoretical peak printed by the CUDA harness. Both are estimates from a traffic model, not hardware counters. Saying the kernel is "memory-bound" is a hypothesis consistent with the numbers; confirming it needs Nsight Compute (`ncu`) DRAM-throughput counters.

## 6. Threats to validity (state these in the report)

* Colab is a shared service: the GPU model, CPU count and clocks are not guaranteed between sessions. All final numbers must come from one session, and the GPU/CPU model in `results/env/environment.json` must be quoted.
* The Colab VM has very few CPU cores, which caps the OpenMP baseline. A GPU speedup measured against a 2-thread CPU is not a statement about a 32-core server.
* The CPU code is plain C compiled by gcc; it is not hand-vectorised or cache-blocked. It is a reasonable but not an optimally tuned baseline.
* `--fmad=false` is used so GPU and CPU round identically. Production CUDA code normally allows FMA, which is slightly faster and changes results in the last bits.
* float32 rounding alone reaches about 6e-5 degC against a float64 reference after 1000 steps (`results/correctness_full.csv`), i.e. only about 1.7x below the 1e-4 acceptance tolerance. CPU-GPU error of exactly 0 means the two round identically, not that either is exact.
* Block-shape differences of a few percent are within run-to-run noise unless the IQR bars clearly do not overlap.
