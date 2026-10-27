#!/usr/bin/env python3
"""
validate_all.py -- full correctness matrix (Person 3, "all three validate" stage).

Compares every implementation against the CPU serial reference (benchmarks/cpu_bench.c) under
IDENTICAL conditions, including cases the single 1024x1024 / 16x16 check does not cover:
  * grid sizes that are NOT a multiple of the block size (1000, 513, 200)
  * a single step (catches indexing / halo bugs immediately) and many steps (accumulating error)
  * every CUDA block shape, for both the naive and the shared-memory tiled kernel
  * float32 CPU vs an independent float64 NumPy reference (how large is plain rounding error?)
  * Person 1's own results/cpu/final_serial.csv vs cpu_bench (cross-validation of the two CPU codes)

Pass criteria (fixed in advance, same as scripts/compare_results.py):
  max abs error <= 1e-4 degC, all values finite, boundary cells exactly 25 degC,
  maximum principle holds (25 <= T <= 100 -- diffusion cannot create new extremes).

Writes results/correctness_full.csv. Exit code 1 if any check fails.

usage: python benchmarks/validate_all.py [--out results] [--skip-cuda] [--quick]
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import reference as ref  # noqa: E402

TOL = 1e-4
SHAPES = [(8, 8), (16, 16), (32, 8), (32, 16), (32, 32)]
CASES_FULL = [(1024, 1000), (1000, 1000), (513, 200), (1024, 1), (200, 50)]
CASES_QUICK = [(513, 100), (200, 1), (200, 50)]


def sh(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def read_grid(path):
    return pd.read_csv(path, header=None, dtype=np.float32).to_numpy()


def build(build_dir, need_cuda):
    build_dir.mkdir(parents=True, exist_ok=True)
    sh(["gcc", "-O3", "-ffp-contract=off", "-fopenmp", "benchmarks/cpu_bench.c", "-o", str(build_dir / "cpu_bench")])
    if need_cuda:
        sh(["nvcc", "-O3", "--fmad=false", "-I", "include", "src/cuda_naive.cu", "-o", str(build_dir / "cuda_naive")])
        sh(["nvcc", "-O3", "--fmad=false", "-I", "include", "src/cuda_optimized.cu", "-o", str(build_dir / "cuda_optimized")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller cases (faster)")
    ap.add_argument("--omp-threads", type=int, default=0, help="threads for the OpenMP check (0 = all)")
    a = ap.parse_args()

    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bd = ROOT / a.build_dir
    have_nvcc = subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0
    use_cuda = (not a.skip_cuda) and have_nvcc
    if not a.skip_cuda and not have_nvcc:
        print("[warn] nvcc not found -> CUDA checks skipped (run this on Colab with a GPU runtime)")
    build(bd, use_cuda)

    rows, all_ok = [], True
    tmp = Path(tempfile.mkdtemp(prefix="validate_"))

    def check(impl, N, K, block, reference_name, test, refgrid, informational=False):
        nonlocal all_ok
        e_max, rmse, ndiff = ref.errors(refgrid, test)
        finite = bool(np.isfinite(test).all())
        bnd = ref.boundary_ok(test)
        mp = bool(test.min() >= ref.AMBIENT - 1e-3 and test.max() <= ref.HOT + 1e-3)
        ok = finite and bnd and mp and (e_max <= (1e-3 if informational else TOL))
        if not informational:
            all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {impl:<20} N={N:<5} K={K:<5} block={block:<6} vs {reference_name:<14} "
              f"Emax={e_max:.2e} RMSE={rmse:.2e} differing={ndiff}")
        rows.append(dict(impl=impl, N=N, K=K, block=block, reference=reference_name,
                         tolerance=1e-3 if informational else TOL, max_abs_error=f"{e_max:.6e}", rmse=f"{rmse:.6e}",
                         differing_cells=ndiff, finite=int(finite), boundary_fixed=int(bnd), max_principle=int(mp),
                         informational=int(informational), passed=int(ok)))

    for N, K in (CASES_QUICK if a.quick else CASES_FULL):
        print(f"\n=== N={N}, K={K}")
        serial_csv = tmp / f"serial_{N}_{K}.csv"
        sh([str(bd / "cpu_bench"), "serial", str(N), str(K), "1", "1", str(serial_csv)])
        serial = read_grid(serial_csv)

        omp_csv = tmp / f"omp_{N}_{K}.csv"
        sh([str(bd / "cpu_bench"), "omp", str(N), str(K), str(a.omp_threads), "1", str(omp_csv)])
        check("cpu_omp", N, K, "-", "cpu_serial", read_grid(omp_csv), serial)

        if N <= 1024 and K >= 50:
            ref64 = ref.simulate(N, K, np.float64)
            check("cpu_serial(float32)", N, K, "-", "numpy float64", serial, ref64, informational=True)

        if use_cuda:
            for impl, exe in (("cuda_naive", "cuda_naive"), ("cuda_tiled_shared", "cuda_optimized")):
                for bx, by in SHAPES:
                    g = tmp / f"{impl}_{N}_{K}_{bx}x{by}.csv"
                    sh([str(bd / exe), str(N), str(K), str(bx), str(by), "1", str(g)], cwd=tmp)
                    check(impl, N, K, f"{bx}x{by}", "cpu_serial", read_grid(g), serial)

    # Person 1's own output vs the benchmark CPU code
    p1 = ROOT / "results" / "cpu" / "final_serial.csv"
    if p1.exists() and not a.quick:
        print("\n=== cross-check: Person 1 src/cpu_serial.c output vs benchmarks/cpu_bench.c (N=1024, K=1000)")
        ours = read_grid(tmp / "serial_1024_1000.csv")
        check("cpu_bench(serial)", 1024, 1000, "-", "P1 serial CSV", ours, read_grid(p1))

    fields = list(rows[0].keys())
    path = out / "correctness_full.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_fail = sum(1 for r in rows if not r["passed"] and not r["informational"])
    print(f"\n{len(rows)} checks written to {path}; failures (excluding informational): {n_fail}")
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
