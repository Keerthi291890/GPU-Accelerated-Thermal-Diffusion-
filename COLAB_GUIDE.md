# Person 2 – Colab runbook (CUDA part)

Files to add to the repo (do NOT overwrite Person 1's `include/simulation_config.h`):

```
include/cuda_common.h
src/cuda_naive.cu
src/cuda_optimized.cu
scripts/compare_results.py
```

## 0. Colab setup
Runtime → Change runtime type → **T4 GPU**. Then, in cells:

```bash
!nvidia-smi
!nvcc --version
!git clone https://github.com/Keerthi291890/GPU-Accelerated-Thermal-Diffusion-.git proj
%cd proj
```
(Copy your 4 new files into `proj/` – upload them, or `git pull` after pushing from your laptop.)

## 1. Build the CPU reference ON COLAB (same machine as the GPU)
Person 1's `.exe` files are Windows binaries and will not run here. Rebuild from source:
```bash
!gcc -O3 src/cpu_serial.c -o cpu_serial
!gcc -O3 -fopenmp src/cpu_openmp.c -o cpu_openmp
!mkdir -p results/cpu && ./cpu_serial          # writes results/cpu/final_serial.csv
```
(`cpu_serial.c` calls a Windows `mkdir` via `system()`; it still works on Linux because I
create the folder first. Ignore the harmless shell message.)

## 2. Build + run the CUDA programs
```bash
!nvcc -O3 --fmad=false -I include src/cuda_naive.cu     -o cuda_naive
!nvcc -O3 --fmad=false -I include src/cuda_optimized.cu -o cuda_optimized

# N  K    bx by reps  output
!./cuda_naive     1024 1000 16 16 5 results/cuda/final_cuda_naive.csv
!./cuda_optimized 1024 1000 16 16 5 results/cuda/final_cuda_tiled.csv
```
Expected: `Center temperature : 71.636833`.

## 3. Validate (do this BEFORE any benchmarking)
```bash
!python scripts/compare_results.py results/cpu/final_serial.csv results/cuda/final_cuda_naive.csv --label cuda_naive_16x16
!python scripts/compare_results.py results/cpu/final_serial.csv results/cuda/final_cuda_tiled.csv --label cuda_tiled_16x16
```
Tolerance is 1e-4 °C (max abs error), fixed in advance. Results are appended to `results/correctness.csv`.

## 4. Block-shape sweep (feeds Person 3's plots)
```bash
for bs in "8 8" "16 16" "32 8" "32 16" "32 32"; do ./cuda_naive 1024 1000 $bs 7; done
for bs in "8 8" "16 16" "32 8" "32 16" "32 32"; do ./cuda_optimized 1024 1000 $bs 7; done
```
Every run appends one row to `results/cuda/runtime_cuda.csv`
(median / min / max kernel time, median end-to-end time, throughput, effective bandwidth).

## 5. Other grid sizes (optional, for Person 3)
Grid size is a runtime argument for CUDA, but `GRID_SIZE` is a compile-time constant in
Person 1's CPU code. Ask Person 1 to change the header to:
```c
#ifndef GRID_SIZE
#define GRID_SIZE 1024
#endif
```
so the CPU baseline can be rebuilt with `gcc -DGRID_SIZE=2048 ...` for each size.
