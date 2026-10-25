#!/usr/bin/env python3
"""
compare_results.py -- validate a GPU result against Person 1's CPU reference.

usage:
    python scripts/compare_results.py results/cpu/final_serial.csv results/cuda/final_cuda_naive.csv
    python scripts/compare_results.py REF.csv TEST.csv --tol 1e-4 --label cuda_naive

Checks (all must pass):
  1. same shape, no NaN / Inf
  2. max absolute error  E_max = max|T_ref - T_test|   <= tolerance
  3. RMSE = sqrt(mean((T_ref - T_test)^2))              (reported)
  4. boundary cells still equal AMBIENT (25) -> fixed boundary condition held
  5. hotspot has spread: peak below 100 and centre cooler than the initial 100

Tolerance (decided BEFORE looking at results): 1e-4 degC.
  float32 has ~7 significant digits; one ulp near 100 degC is ~7.6e-6.  Compiled with
  --fmad=false and the same operation order the GPU is expected to match the CPU almost
  exactly (often 0).  1e-4 leaves room for a few ulps of rounding difference but would
  expose any real bug (wrong index, in-place update, missing halo) which gives errors
  of order 1e-1 to 1e+1.

Appends a row to results/correctness.csv (for Person 3's report).
"""
import argparse
import csv
import os
import sys

import numpy as np

AMBIENT = 25.0


def load(path):
    try:
        import pandas as pd
        return pd.read_csv(path, header=None, dtype=np.float32).to_numpy()
    except ImportError:
        return np.loadtxt(path, delimiter=",", dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--tol", type=float, default=1e-4, help="max-abs-error tolerance (default 1e-4)")
    ap.add_argument("--label", default=None, help="name for the results row (default: test file name)")
    ap.add_argument("--out", default="results/correctness.csv")
    a = ap.parse_args()

    ref = load(a.reference).astype(np.float64)
    tst = load(a.test).astype(np.float64)

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        ok = ok and bool(cond)

    print(f"Reference: {a.reference}\nTest     : {a.test}")
    check(ref.shape == tst.shape, f"shapes match {ref.shape} vs {tst.shape}")
    if ref.shape != tst.shape:
        sys.exit(1)
    check(np.isfinite(tst).all(), "no NaN / Inf in test result")

    diff = np.abs(ref - tst)
    e_max = float(diff.max())
    rmse = float(np.sqrt(np.mean((ref - tst) ** 2)))
    n_diff = int(np.count_nonzero(diff))
    iy, ix = np.unravel_index(int(diff.argmax()), diff.shape)

    check(e_max <= a.tol, f"max abs error {e_max:.3e} <= tolerance {a.tol:.0e}")
    print(f"         RMSE = {rmse:.3e}   differing cells = {n_diff} / {diff.size}   worst cell = (row {iy}, col {ix})")

    edge = np.concatenate([tst[0, :], tst[-1, :], tst[:, 0], tst[:, -1]])
    check(np.allclose(edge, AMBIENT, atol=0.0), "boundary cells unchanged (= 25 degC)")

    n = tst.shape[0]
    c = n // 2
    check(tst.max() < 100.0, f"hotspot spread: peak = {tst.max():.4f} < 100")
    check(tst[c, c] < 100.0 and tst[c, c] > AMBIENT, f"centre cell = {tst[c, c]:.6f} (between 25 and 100)")
    print(f"         centre: reference {ref[c, c]:.6f}  test {tst[c, c]:.6f}")

    label = a.label or os.path.basename(a.test)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "grid_size", "tolerance", "max_abs_error", "rmse", "differing_cells", "passed"])
        w.writerow([label, n, a.tol, f"{e_max:.6e}", f"{rmse:.6e}", n_diff, int(ok)])

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
