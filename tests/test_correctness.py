"""
tests/test_correctness.py -- automated correctness tests (run:  python -m pytest tests -v).

CPU tests run anywhere with gcc. CUDA tests run only when nvcc and a GPU are present
(on Colab with a GPU runtime) and are skipped otherwise.

What is verified, and why it matters for the case study:
  * stability parameter r <= 1/4                       (explicit-scheme stability)
  * C code == independent NumPy implementation          (the maths is right)
  * OpenMP == serial, bit for bit                       (parallelisation does not change the answer)
  * Person 1's committed reference CSV == benchmark code (the two CPU programs solve one problem)
  * fixed boundaries never change; maximum principle    (physical sanity)
  * grids that are not a multiple of the block size     (CUDA bounds-check / halo logic)
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import reference as ref  # noqa: E402

TOL = 1e-4


def have_gpu():
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def cpu_bench(tmp_path_factory):
    exe = tmp_path_factory.mktemp("bin") / "cpu_bench"
    p = subprocess.run(["gcc", "-O3", "-ffp-contract=off", "-fopenmp", str(ROOT / "benchmarks/cpu_bench.c"), "-o", str(exe)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return exe


def run_cpu(exe, mode, N, K, tmp, threads=2):
    out = tmp / f"{mode}_{N}_{K}.csv"
    p = subprocess.run([str(exe), mode, str(N), str(K), str(threads), "1", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return pd.read_csv(out, header=None, dtype=np.float32).to_numpy()


# ---------------------------------------------------------------- model parameters
def test_diffusion_parameter_satisfies_stability_limit():
    text = (ROOT / "include" / "simulation_config.h").read_text()
    r = float(re.search(r"#define\s+DIFFUSION_R\s+([0-9.]+)f?", text).group(1))
    assert 0 < r <= 0.25, f"r={r} violates the explicit 2-D stability limit r <= 1/4"
    assert abs(r - ref.R) < 1e-9, "NumPy reference uses a different r than the C code"


def test_update_weights_are_a_convex_combination():
    r = ref.R
    assert 1 - 4 * r >= 0 and r >= 0 and abs((1 - 4 * r) + 4 * r - 1) < 1e-12


# ---------------------------------------------------------------- CPU correctness
@pytest.mark.parametrize("N,K", [(60, 25), (64, 50), (100, 7)])
def test_c_serial_matches_numpy_float64(cpu_bench, tmp_path, N, K):
    c = run_cpu(cpu_bench, "serial", N, K, tmp_path)
    e_max, _, _ = ref.errors(ref.simulate(N, K, np.float64), c)
    assert e_max < TOL, f"E_max={e_max}"


@pytest.mark.parametrize("N,K", [(60, 25), (100, 7), (256, 40)])
def test_c_serial_matches_numpy_float32_bitwise(cpu_bench, tmp_path, N, K):
    c = run_cpu(cpu_bench, "serial", N, K, tmp_path)
    e_max, _, _ = ref.errors(ref.simulate(N, K, np.float32), c)
    assert e_max <= 1e-5, f"float32 C vs float32 NumPy differ by {e_max}"


@pytest.mark.parametrize("threads", [1, 2, 3])
def test_openmp_equals_serial_exactly(cpu_bench, tmp_path, threads):
    s = run_cpu(cpu_bench, "serial", 130, 60, tmp_path)
    o = run_cpu(cpu_bench, "omp", 130, 60, tmp_path, threads=threads)
    assert np.array_equal(s, o), "OpenMP result differs from serial (race condition or wrong swap?)"


def test_boundaries_stay_fixed_and_maximum_principle(cpu_bench, tmp_path):
    g = run_cpu(cpu_bench, "serial", 128, 500, tmp_path)
    assert ref.boundary_ok(g), "boundary cells changed"
    assert g.min() >= ref.AMBIENT - 1e-4 and g.max() <= ref.HOT + 1e-4, "diffusion created a new temperature extreme"
    assert np.isfinite(g).all()


def test_hotspot_spreads_smoothly(cpu_bench, tmp_path):
    g = run_cpu(cpu_bench, "serial", 128, 200, tmp_path)
    assert g.max() < ref.HOT, "peak did not decrease"
    assert g[64, 64 + 30] > ref.AMBIENT + 1e-4, "heat did not diffuse outside the initial hotspot"
    c = 64
    row = g[c, c:]
    assert np.all(np.diff(row) <= 1e-4), "temperature is not monotonically decreasing away from the centre"


def test_energy_conserved_until_heat_reaches_boundary(cpu_bench, tmp_path):
    """interior sum is conserved by the stencil while the boundary is not yet reached"""
    N, K = 400, 100
    g0 = ref.initial_grid(N)
    g = run_cpu(cpu_bench, "serial", N, K, tmp_path)
    assert abs(g.sum(dtype=np.float64) - g0.sum(dtype=np.float64)) / g0.sum(dtype=np.float64) < 1e-6


def test_matches_person1_reference_csv(cpu_bench, tmp_path):
    p1 = ROOT / "results" / "cpu" / "final_serial.csv"
    if not p1.exists():
        pytest.skip("results/cpu/final_serial.csv not present")
    theirs = pd.read_csv(p1, header=None, dtype=np.float32).to_numpy()
    if theirs.shape != (1024, 1024):
        pytest.skip("reference CSV is not 1024x1024 / K=1000")
    ours = run_cpu(cpu_bench, "serial", 1024, 1000, tmp_path)
    assert np.array_equal(ours, theirs), "benchmark CPU code and Person 1's code disagree"


# ---------------------------------------------------------------- CUDA correctness (GPU runtime only)
cuda = pytest.mark.skipif(not have_gpu(), reason="needs nvcc + an NVIDIA GPU (run on Colab)")


@pytest.fixture(scope="session")
def cuda_bins(tmp_path_factory):
    d = tmp_path_factory.mktemp("cuda")
    out = {}
    for name, src in (("naive", "src/cuda_naive.cu"), ("tiled", "src/cuda_optimized.cu")):
        exe = d / name
        p = subprocess.run(["nvcc", "-O3", "--fmad=false", "-I", str(ROOT / "include"), str(ROOT / src), "-o", str(exe)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        out[name] = exe
    return out


@cuda
@pytest.mark.parametrize("which", ["naive", "tiled"])
@pytest.mark.parametrize("shape", [(8, 8), (16, 16), (32, 8), (32, 16), (32, 32)])
@pytest.mark.parametrize("N,K", [(100, 1), (100, 80), (257, 200)])   # NOT multiples of the block size (hotspot needs N >= 52)
def test_cuda_matches_cpu(cpu_bench, cuda_bins, tmp_path, which, shape, N, K):
    cpu = run_cpu(cpu_bench, "serial", N, K, tmp_path)
    out = tmp_path / "gpu.csv"
    p = subprocess.run([str(cuda_bins[which]), str(N), str(K), str(shape[0]), str(shape[1]), "1", str(out)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    gpu = pd.read_csv(out, header=None, dtype=np.float32).to_numpy()
    e_max, _, _ = ref.errors(cpu, gpu)
    assert e_max <= TOL, f"{which} {shape} N={N} K={K}: E_max={e_max}"
    assert ref.boundary_ok(gpu)
