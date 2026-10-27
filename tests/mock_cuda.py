#!/usr/bin/env python3
"""
mock_cuda.py -- a FAKE stand-in for the cuda_naive / cuda_optimized binaries.

It prints output in exactly the same format as include/cuda_common.h so that
benchmarks/run_benchmarks.py and benchmarks/analyze.py can be tested without a GPU.
ALL NUMBERS ARE SYNTHETIC (simple roofline-style model + noise). Never use them in the report.
The device name contains 'SYNTHETIC-MOCK' and analyze.py prints a warning banner for it.

usage: mock_cuda.py --impl cuda_naive|cuda_tiled_shared N K bx by reps
"""
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True)
ap.add_argument("N", type=int); ap.add_argument("K", type=int)
ap.add_argument("bx", type=int); ap.add_argument("by", type=int); ap.add_argument("reps", type=int)
a = ap.parse_args()

PEAK = 320.0           # GB/s, pretend T4
EFF = 0.68             # fraction of peak the naive kernel reaches
LAUNCH = 5.5e-6        # s per kernel launch
rng = random.Random(a.N * 7919 + a.K * 31 + a.bx * 13 + a.by)

bytes_per_step = 8.0 * a.N * a.N
step = max(LAUNCH, bytes_per_step / (PEAK * 1e9 * EFF))
shape_pen = {(8, 8): 1.20, (16, 16): 1.01, (32, 8): 1.0, (32, 16): 1.03, (32, 32): 1.04}.get((a.bx, a.by), 1.1)
tile_pen = 1.0 if a.impl == "cuda_naive" else {(8, 8): 1.14, (16, 16): 1.13, (32, 8): 1.04, (32, 16): 1.10, (32, 32): 1.13}.get((a.bx, a.by), 1.1)
kernel = a.K * step * shape_pen * tile_pen
xfer = 3 * 4.0 * a.N * a.N / 12e9 + 150e-6      # 2 H2D + 1 D2H at ~12 GB/s

print("---------------- GPU device information ----------------")
print("Device                : SYNTHETIC-MOCK-GPU (not real)")
print("Compute capability    : 7.5")
print("Streaming MPs (SMs)   : 40")
print("Theoretical peak BW   : %.1f GB/s" % PEAK)
print("--------------------------------------------------------")
print("Implementation : %s" % a.impl)
for i in range(a.reps):
    k = kernel * (1 + abs(rng.gauss(0, 0.006)))
    print("  run %2d: kernel-only %.6f s   end-to-end %.6f s" % (i + 1, k, k + xfer))
print("Center temperature : 71.636833")
