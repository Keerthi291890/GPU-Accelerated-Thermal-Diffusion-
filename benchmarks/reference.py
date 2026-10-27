"""
reference.py -- independent NumPy reference for the heat-diffusion benchmark.

Used by tests/test_correctness.py and benchmarks/validate_all.py to check the C / CUDA programs
against a THIRD implementation (vectorised NumPy, two-buffer, same initial grid and fixed
boundary). Running it in float64 also shows how much of any CPU/GPU difference is just
float32 rounding.
"""
import numpy as np

AMBIENT, HOT, RADIUS, R = 25.0, 100.0, 25, 0.2


def initial_grid(N, dtype=np.float32):
    g = np.full((N, N), AMBIENT, dtype=dtype)
    c = N // 2
    g[c - RADIUS:c + RADIUS, c - RADIUS:c + RADIUS] = HOT
    return g


def simulate(N, K, dtype=np.float32, r=R):
    """K explicit steps of the 5-point heat stencil; boundary cells never change."""
    cur = initial_grid(N, dtype)
    nxt = cur.copy()
    r = dtype(r)
    four = dtype(4.0)
    for _ in range(K):
        c = cur[1:-1, 1:-1]
        # same operation order as the C/CUDA code:  c + r*(up + down + left + right - 4c)
        nxt[1:-1, 1:-1] = c + r * (cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:] - four * c)
        cur, nxt = nxt, cur
    return cur


def errors(ref, test):
    d = np.abs(ref.astype(np.float64) - test.astype(np.float64))
    return float(d.max()), float(np.sqrt(np.mean(d ** 2))), int(np.count_nonzero(d))


def boundary_ok(g, value=AMBIENT):
    edge = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    return bool(np.all(edge == value))
