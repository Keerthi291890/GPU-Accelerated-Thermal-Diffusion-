
# GPU-Accelerated Thermal Diffusion

A high-performance computing case study investigating CPU parallelism,
CUDA acceleration, memory bandwidth, and cloud GPU execution through
a two-dimensional thermal-diffusion simulation.

## Research Question

How do CPU parallelism, CUDA thread configuration, memory-access
patterns, and problem size affect runtime, speedup, and numerical
correctness?

## Simulation

We model heat diffusion across a two-dimensional plate using an
explicit finite-difference approximation of the heat equation.

The initial plate has a central hotspot and fixed-temperature
boundaries.

## Implementations

- Serial C++ CPU baseline
- OpenMP parallel CPU implementation
- Naive CUDA implementation
- Shared-memory CUDA optimization, if feasible

## Evaluation

- Runtime and speedup
- CPU and GPU correctness
- Maximum absolute error and RMSE
- Cell-update throughput
- Estimated effective memory bandwidth
- Effect of grid size and CUDA block configuration
- Cloud GPU execution and reproducibility

## Repository Structure

- `src/`: CPU and CUDA implementations
- `docs/`: Mathematical model and experiment methodology
- `tests/`: Correctness checks
- `scripts/`: Visualization and utility scripts
- `benchmarks/`: Benchmark orchestration and analysis
- `results/`: Generated outputs and experimental results

## Team Responsibilities

- Person 1: Mathematical model and CPU implementations
- Person 2: CUDA kernels and GPU optimization
- Person 3: Benchmarking, analysis, and cloud HPC


