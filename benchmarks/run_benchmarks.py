#!/usr/bin/env python3
"""
run_benchmarks.py -- Person 3's benchmark orchestrator.

Builds the CPU / CUDA programs, captures the environment, then runs every experiment
and appends ONE ROW PER TIMED REPETITION to results/raw_timings.csv.

Experiments (all use float32, r = 0.2, identical initial grid):
  scaling  A/D : every implementation over a range of grid sizes (K fixed)
  blocks   B   : CUDA block shapes (naive and tiled) at a few grid sizes
  steps    D   : number of time steps K swept at one grid size -> shows launch +
                 transfer overhead being amortised (kernel-only vs end-to-end)
  threads      : OpenMP thread count at one grid size (CPU parallel scaling)

Fair-measurement rules built in
  * warm-up inside every program (not timed)
  * ROUNDS: every configuration is run `--rounds` times and the job order is shuffled
    each round, so slow GPU clock drift / noisy neighbours do not bias one method
  * CUDA programs time kernels with cudaEvents (kernel-only AND end-to-end incl. PCIe)
  * resumable: re-running skips rows already in raw_timings.csv (Colab sessions die)

Usage (from the repository root):
  python benchmarks/run_benchmarks.py --env-only
  python benchmarks/run_benchmarks.py                      # full study
  python benchmarks/run_benchmarks.py --only scaling       # one experiment
  python benchmarks/run_benchmarks.py --sizes 512,1024 --rounds 1   # quick smoke test
  python benchmarks/run_benchmarks.py --mock --out results/mock     # test pipeline, NO GPU needed
"""
import argparse
import csv
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TILED_SHAPES = {(8, 8), (16, 16), (32, 8), (32, 16), (32, 32)}   # supported by src/cuda_optimized.cu
FIELDS = ["timestamp", "impl", "N", "K", "block_x", "block_y", "threads", "round", "rep",
          "kernel_s", "e2e_s", "device", "peak_bw_GBps"]

CPU_FLAGS = ["-O3", "-ffp-contract=off", "-fopenmp"]
NVCC_FLAGS = ["-O3", "--fmad=false", "-I", "include"]


# ----------------------------------------------------------------------------- helpers
def sh(cmd, cwd=ROOT, timeout=None):
    """run a command, return (returncode, stdout+stderr)"""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_shapes(s):
    out = []
    for tok in s.split(","):
        bx, by = tok.lower().split("x")
        out.append((int(bx), int(by)))
    return out


def parse_ints(s):
    return [int(t) for t in s.split(",") if t.strip()]


# ----------------------------------------------------------------------------- environment
def capture_environment(out_dir, build_info):
    env_dir = Path(out_dir) / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    cmds = {
        "nvidia_smi.txt": ["nvidia-smi"],
        "nvidia_smi_query.csv": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit",
                                 "--format=csv"],
        "nvcc_version.txt": ["nvcc", "--version"],
        "gcc_version.txt": ["gcc", "--version"],
        "lscpu.txt": ["lscpu"],
    }
    for fname, cmd in cmds.items():
        rc, out = sh(cmd)
        (env_dir / fname).write_text(out if rc == 0 else f"[unavailable: rc={rc}] {out}")

    def first_line(fname, default=""):
        t = (env_dir / fname).read_text().strip().splitlines()
        return t[0] if t and not t[0].startswith("[unavailable") else default

    gpu_name = ""
    q = (env_dir / "nvidia_smi_query.csv").read_text().strip().splitlines()
    if len(q) > 1 and not q[0].startswith("[unavailable"):
        gpu_name = q[1].split(",")[0].strip()
    cpu_model = ""
    for line in (env_dir / "lscpu.txt").read_text().splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
    nvcc = ""
    for line in (env_dir / "nvcc_version.txt").read_text().splitlines():
        if "release" in line:
            nvcc = line.strip()
    _, commit = sh(["git", "rev-parse", "--short", "HEAD"])
    info = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu_name": gpu_name or "none detected",
        "cpu_model": cpu_model or platform.processor() or "unknown",
        "cpu_threads_available": ncpu(),
        "nvcc": nvcc or "not available",
        "gcc": first_line("gcc_version.txt", "not available"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "git_commit": commit.strip(),
        "cpu_flags": " ".join(CPU_FLAGS),
        "nvcc_flags": " ".join(NVCC_FLAGS),
        "precision": "float32",
        "diffusion_r": 0.2,
    }
    info.update(build_info or {})
    (env_dir / "environment.json").write_text(json.dumps(info, indent=2))
    return info


# ----------------------------------------------------------------------------- build
def build_all(build_dir, rebuild, need_cpu, need_cuda):
    build_dir.mkdir(parents=True, exist_ok=True)
    bins = {}
    if need_cpu:
        exe = build_dir / "cpu_bench"
        if rebuild or not exe.exists():
            print("[build] cpu_bench")
            rc, out = sh(["gcc", *CPU_FLAGS, "benchmarks/cpu_bench.c", "-o", str(exe)])
            if rc != 0:
                sys.exit(f"cpu_bench build failed:\n{out}")
        bins["cpu_bench"] = exe
    if need_cuda:
        for name, src in (("cuda_naive", "src/cuda_naive.cu"), ("cuda_optimized", "src/cuda_optimized.cu")):
            exe = build_dir / name
            if rebuild or not exe.exists():
                print(f"[build] {name}")
                rc, out = sh(["nvcc", *NVCC_FLAGS, src, "-o", str(exe)])
                if rc != 0:
                    sys.exit(f"{name} build failed (is nvcc installed / GPU runtime selected?):\n{out}")
            bins[name] = exe
    return bins


# ----------------------------------------------------------------------------- job list
def make_jobs(a):
    """Return a de-duplicated list of dict jobs."""
    jobs = {}

    def add(impl, N, K, bx=0, by=0, threads=1):
        if impl.startswith("cuda") and a.skip_cuda:
            return
        if impl.startswith("cpu") and a.skip_cpu:
            return
        if impl == "cuda_tiled_shared" and (bx, by) not in TILED_SHAPES:
            return
        jobs[(impl, N, K, bx, by, threads)] = dict(impl=impl, N=N, K=K, bx=bx, by=by, threads=threads)

    nthr = ncpu() if a.threads == "auto" else int(a.threads)
    only = set(a.only.split(",")) if a.only else {"scaling", "blocks", "steps", "threads"}

    if "scaling" in only:
        for N in parse_ints(a.sizes):
            add("cpu_serial", N, a.K)
            add("cpu_omp", N, a.K, threads=nthr)
            for bx, by in parse_shapes(a.scaling_blocks):
                add("cuda_naive", N, a.K, bx, by)
                add("cuda_tiled_shared", N, a.K, bx, by)
    if "blocks" in only:
        for N in parse_ints(a.block_grids):
            for bx, by in parse_shapes(a.blocks):
                add("cuda_naive", N, a.K, bx, by)
                add("cuda_tiled_shared", N, a.K, bx, by)
    if "steps" in only:
        bx, by = parse_shapes(a.step_block)[0]
        for K in parse_ints(a.step_sweep):
            add("cpu_serial", a.step_grid, K)
            add("cpu_omp", a.step_grid, K, threads=nthr)
            add("cuda_naive", a.step_grid, K, bx, by)
    if "threads" in only:
        tl = sorted({t for t in parse_ints(a.thread_list) if t >= 1} | {nthr})
        for t in tl:
            add("cpu_omp", a.thread_grid, a.K, threads=t)
        add("cpu_serial", a.thread_grid, a.K)
    return list(jobs.values())


# ----------------------------------------------------------------------------- running + parsing
RE_DEVICE = re.compile(r"^Device\s*:\s*(.+)$", re.M)
RE_PEAK = re.compile(r"Theoretical peak BW\s*:\s*([\d.]+)")
RE_RUN = re.compile(r"run\s+(\d+):\s+kernel-only\s+([\d.]+)\s+s\s+end-to-end\s+([\d.]+)\s+s")
RE_CPU = re.compile(r"^RESULT\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)$", re.M)


def run_job(job, bins, a, scratch):
    impl, N, K = job["impl"], job["N"], job["K"]
    rows = []
    if impl.startswith("cpu"):
        mode = "serial" if impl == "cpu_serial" else "omp"
        cmd = [str(bins["cpu_bench"]), mode, str(N), str(K), str(job["threads"]), "1"]
        rc, out = sh(cmd, cwd=ROOT)
        if rc != 0:
            return None, out
        for m in RE_CPU.finditer(out):
            t = float(m.group(6))
            rows.append(dict(kernel_s=t, e2e_s=t, device="CPU", peak_bw_GBps=""))
        return rows, out

    if a.mock:
        prefix = [sys.executable, str(ROOT / "tests" / "mock_cuda.py"), "--impl", impl]
    else:
        prefix = [str(bins["cuda_naive" if impl == "cuda_naive" else "cuda_optimized"])]
    cmd_args = [
    str(N), str(K), str(job["bx"]),
    str(job["by"]), str(a.cuda_reps)]

# The real CUDA harness accepts an optional output-file path.
# The mock program does not accept this extra argument.
    if not a.mock:
      cmd_args.append(str(scratch / "cuda_harness_output.csv"))

    cmd = prefix + cmd_args
    rc, out = sh(cmd, cwd=scratch)

    if rc != 0:
        return None, out
    dev = RE_DEVICE.search(out)
    peak = RE_PEAK.search(out)
    for m in RE_RUN.finditer(out):
        rows.append(dict(kernel_s=float(m.group(2)), e2e_s=float(m.group(3)),
                         device=dev.group(1).strip() if dev else "unknown",
                         peak_bw_GBps=peak.group(1) if peak else ""))
    return rows, out


def load_done(path):
    done = set()
    if path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["impl"], int(r["N"]), int(r["K"]), int(r["block_x"]), int(r["block_y"]),
                          int(r["threads"]), int(r["round"])))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results", help="output directory (default results)")
    p.add_argument("--build-dir", default="build")
    p.add_argument("--rounds", type=int, default=3, help="times each config is run (job order shuffled per round)")
    p.add_argument("--cuda-reps", type=int, default=7, help="timed repetitions inside each CUDA process")
    p.add_argument("--K", type=int, default=1000, help="time steps for the size / block / thread experiments")
    p.add_argument("--sizes", default="128,256,512,1024,2048,4096", help="grid sizes for the scaling experiment (add 8192 if memory/time allow)")
    p.add_argument("--scaling-blocks", default="16x16,32x8")
    p.add_argument("--blocks", default="8x8,16x16,32x8,32x16,32x32")
    p.add_argument("--block-grids", default="1024,4096")
    p.add_argument("--step-sweep", default="10,100,1000,10000")
    p.add_argument("--step-grid", type=int, default=1024)
    p.add_argument("--step-block", default="32x8")
    p.add_argument("--thread-grid", type=int, default=2048)
    p.add_argument("--thread-list", default="1,2,4")
    p.add_argument("--threads", default="auto", help="OpenMP threads for the main comparison (auto = all available)")
    p.add_argument("--only", default="", help="comma list of: scaling,blocks,steps,threads")
    p.add_argument("--skip-cuda", action="store_true")
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--mock", action="store_true", help="use tests/mock_cuda.py instead of real CUDA (pipeline test only; data is SYNTHETIC)")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--env-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    out_dir = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = ROOT / a.build_dir
    scratch = out_dir / "_cuda_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    need_cuda = not (a.skip_cuda or a.mock or a.env_only or a.dry_run)
    need_cpu = not (a.skip_cpu or a.env_only or a.dry_run)
    bins = build_all(build_dir, a.rebuild, need_cpu, need_cuda) if (need_cpu or need_cuda) else {}

    info = capture_environment(out_dir, {"mock": a.mock})
    print(f"[env] GPU: {info['gpu_name']} | CPU: {info['cpu_model']} ({info['cpu_threads_available']} threads) | {info['nvcc']}")
    if a.env_only:
        print(f"[env] saved to {out_dir/'env'}")
        return

    jobs = make_jobs(a)
    total = len(jobs) * a.rounds
    print(f"[plan] {len(jobs)} configurations x {a.rounds} rounds = {total} process runs")
    if a.dry_run:
        for j in jobs:
            print("   ", j)
        return
    if not a.skip_cuda and not a.mock and info["gpu_name"] == "none detected":
        sys.exit("No GPU detected. In Colab: Runtime -> Change runtime type -> T4 GPU. (Use --skip-cuda for CPU-only runs.)")

    raw = out_dir / "raw_timings.csv"
    done = load_done(raw)
    new_file = not raw.exists()
    fails = out_dir / "benchmark_failures.log"
    f = raw.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
    if new_file:
        w.writeheader()

    rng = random.Random(a.seed)
    t_start = time.time()
    n_run = 0
    for rnd in range(1, a.rounds + 1):
        order = jobs[:]
        rng.shuffle(order)                  # interleave implementations / sizes within a round
        for job in order:
            key = (job["impl"], job["N"], job["K"], job["bx"], job["by"], job["threads"], rnd)
            if key in done:
                continue
            n_run += 1
            label = f"{job['impl']:<18} N={job['N']:<5} K={job['K']:<6} blk={job['bx']}x{job['by']:<3} thr={job['threads']}"
            rows, out = run_job(job, bins, a, scratch)
            if not rows:
                print(f"  [round {rnd}] {label}  FAILED (see {fails.name})")
                with fails.open("a") as ff:
                    ff.write(f"=== {label} round {rnd}\n{out}\n")
                continue
            for rep, r in enumerate(rows, 1):
                w.writerow(dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                impl=job["impl"], N=job["N"], K=job["K"], block_x=job["bx"], block_y=job["by"],
                                threads=job["threads"], round=rnd, rep=rep, **r))
            f.flush()
            med = sorted(r["kernel_s"] for r in rows)[len(rows) // 2]
            el = time.time() - t_start
            print(f"  [round {rnd}] {label}  median {med*1e3:10.3f} ms   (elapsed {el:5.0f}s)")
    f.close()
    print(f"[done] {n_run} runs this session -> {raw}")
    if fails.exists():
        print(f"[warn] some runs failed -- see {fails}")


if __name__ == "__main__":
    main()
