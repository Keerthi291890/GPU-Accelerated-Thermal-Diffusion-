#!/usr/bin/env python3
"""
analyze.py -- turn results/raw_timings.csv into metrics, tables and plots (Person 3).

Reads : <out>/raw_timings.csv, <out>/env/environment.json
Writes: <out>/summary.csv            every configuration with all metrics
        <out>/summary_tables.md      report-ready tables + automatic fact checks
        <out>/plots/*.png            figures for the report / slides

Metric definitions (state these in the report):
  kernel time   median over ALL repetitions of the cudaEvent-timed K-step loop (GPU) or the
                wall-clock K-step loop (CPU). No transfers, no allocation, no file I/O.
  end-to-end    GPU: 2 host->device copies + K kernels + 1 device->host copy (cudaEvents).
                CPU: identical to kernel time (data already in host RAM, nothing to transfer).
  updates       U = (N-2)^2 * K  interior-cell updates
  throughput    U / kernel time                           [Gcell-updates/s]
  speedup       T_baseline / T_gpu, serial and OpenMP baselines reported SEPARATELY
  runtime red.  (T_cpu - T_gpu) / T_cpu * 100 %
  bandwidth     8 B/update  (1 read + 1 write of float32, perfect neighbour reuse)  <- main model
                24 B/update (5 reads + 1 write, NO reuse) <- naive-traffic model; can exceed the
                physical peak because caches serve most of those reads. Neither is measured DRAM traffic.
  % of peak     estimated bandwidth / theoretical peak printed by the CUDA harness
  opt. speedup  T_naive / T_tiled at identical (N, K, block shape)

usage: python benchmarks/analyze.py [--out results]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["impl", "N", "K", "block_x", "block_y", "threads"]
COLORS = {"cpu_serial": "#6b6b6b", "cpu_omp": "#1f77b4", "cuda_naive": "#2ca02c", "cuda_tiled_shared": "#ff7f0e"}
LABELS = {"cpu_serial": "CPU serial (C, -O3)", "cpu_omp": "CPU OpenMP", "cuda_naive": "CUDA naive", "cuda_tiled_shared": "CUDA shared-memory tiled"}
SHAPE_ORDER = [(8, 8), (16, 16), (32, 8), (32, 16), (32, 32)]


# ------------------------------------------------------------------ load + aggregate
def load(out_dir):
    raw = pd.read_csv(out_dir / "raw_timings.csv")
    env_path = out_dir / "env" / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    return raw, env


def aggregate(raw):
    q1 = lambda s: s.quantile(0.25)
    q3 = lambda s: s.quantile(0.75)
    agg = raw.groupby(KEYS).agg(
        n=("kernel_s", "size"), rounds=("round", "nunique"),
        kernel_med=("kernel_s", "median"), kernel_q1=("kernel_s", q1), kernel_q3=("kernel_s", q3),
        kernel_min=("kernel_s", "min"), kernel_max=("kernel_s", "max"),
        e2e_med=("e2e_s", "median"), device=("device", "first")).reset_index()
    per_round = raw.groupby(KEYS + ["round"])["kernel_s"].median().reset_index()
    spread = per_round.groupby(KEYS)["kernel_s"].agg(lambda s: (s.max() - s.min())).reset_index(name="round_range_s")
    agg = agg.merge(spread, on=KEYS)
    agg["round_spread_pct"] = 100 * agg["round_range_s"] / agg["kernel_med"]
    agg["iqr_pct"] = 100 * (agg["kernel_q3"] - agg["kernel_q1"]) / agg["kernel_med"]
    return agg


def add_metrics(agg, env, raw):
    peak = pd.to_numeric(raw["peak_bw_GBps"], errors="coerce").dropna()
    peak = float(peak.median()) if len(peak) else np.nan
    nthr = int(env.get("cpu_threads_available", 0) or 0)
    is_gpu = agg["impl"].str.startswith("cuda")
    agg["updates"] = (agg["N"] - 2.0) ** 2 * agg["K"]
    agg["throughput_Gcells_s"] = agg["updates"] / agg["kernel_med"] / 1e9
    agg["transfer_ms"] = np.where(is_gpu, (agg["e2e_med"] - agg["kernel_med"]) * 1e3, 0.0)
    agg["transfer_pct_of_e2e"] = np.where(is_gpu, 100 * (agg["e2e_med"] - agg["kernel_med"]) / agg["e2e_med"], 0.0)
    agg["bw_8B_GBps"] = np.where(is_gpu, 8.0 * agg["throughput_Gcells_s"], np.nan)
    agg["bw_24B_GBps"] = np.where(is_gpu, 24.0 * agg["throughput_Gcells_s"], np.nan)
    agg["pct_peak_8B"] = 100 * agg["bw_8B_GBps"] / peak
    agg["pct_peak_24B"] = 100 * agg["bw_24B_GBps"] / peak

    # CPU baselines per (N,K)
    ser = agg[agg.impl == "cpu_serial"].set_index(["N", "K"])
    omp_rows = agg[agg.impl == "cpu_omp"]
    omp = {}
    for (N, K), grp in omp_rows.groupby(["N", "K"]):
        pick = grp[grp.threads == nthr]
        omp[(N, K)] = (pick if len(pick) else grp.sort_values("kernel_med")).iloc[0]
    for col in ["spd_kernel_vs_serial", "spd_kernel_vs_omp", "spd_e2e_vs_serial", "spd_e2e_vs_omp",
                "runtime_red_vs_serial_pct", "runtime_red_vs_omp_pct", "omp_threads_baseline"]:
        agg[col] = np.nan
    for i, r in agg.iterrows():
        k = (r.N, r.K)
        if r.impl.startswith("cuda"):
            if k in ser.index:
                s = ser.loc[k, "kernel_med"]
                agg.at[i, "spd_kernel_vs_serial"] = s / r.kernel_med
                agg.at[i, "spd_e2e_vs_serial"] = s / r.e2e_med
                agg.at[i, "runtime_red_vs_serial_pct"] = 100 * (s - r.kernel_med) / s
            if k in omp:
                o = omp[k]["kernel_med"]
                agg.at[i, "spd_kernel_vs_omp"] = o / r.kernel_med
                agg.at[i, "spd_e2e_vs_omp"] = o / r.e2e_med
                agg.at[i, "runtime_red_vs_omp_pct"] = 100 * (o - r.kernel_med) / o
                agg.at[i, "omp_threads_baseline"] = omp[k]["threads"]
        elif r.impl == "cpu_omp" and k in ser.index:
            agg.at[i, "spd_kernel_vs_serial"] = ser.loc[k, "kernel_med"] / r.kernel_med

    # optimisation speedup: naive / tiled at identical (N,K,block)
    nv = agg[agg.impl == "cuda_naive"].set_index(["N", "K", "block_x", "block_y"])["kernel_med"]
    agg["opt_speedup_naive_over_tiled"] = np.nan
    for i, r in agg[agg.impl == "cuda_tiled_shared"].iterrows():
        k = (r.N, r.K, r.block_x, r.block_y)
        if k in nv.index:
            agg.at[i, "opt_speedup_naive_over_tiled"] = nv.loc[k] / r.kernel_med
    return agg, peak


def best_shape(agg, impl, N, K):
    sub = agg[(agg.impl == impl) & (agg.N == N) & (agg.K == K)]
    return None if sub.empty else sub.sort_values("kernel_med").iloc[0]


# ------------------------------------------------------------------ plotting helpers
class Plotter:
    def __init__(self, plot_dir, synthetic):
        self.dir = plot_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.synthetic = synthetic

    def finish(self, fig, name):
        if self.synthetic:
            fig.text(0.5, 0.5, "SYNTHETIC GPU DATA\npipeline test only", ha="center", va="center",
                     fontsize=34, color="red", alpha=0.18, rotation=25)
        fig.tight_layout()
        fig.savefig(self.dir / name, dpi=160)
        plt.close(fig)
        print(f"  plot: {name}")


def errbars(df, col="kernel_med", scale=1.0):
    return [(df[col] - df["kernel_q1"]) * scale, (df["kernel_q3"] - df[col]) * scale]


def plots(agg, env, peak, K_main, P):
    gpu_name = env.get("gpu_name", "GPU")
    sub_title = f"{gpu_name} | {env.get('cpu_model','CPU')} x{env.get('cpu_threads_available','?')} threads | float32"

    # best-shape series per N for the main K
    Ns = sorted(agg[(agg.K == K_main)]["N"].unique())

    def series(impl):
        rows = [best_shape(agg, impl, N, K_main) for N in Ns]
        return pd.DataFrame([r for r in rows if r is not None])

    # --- 1 runtime vs size
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for impl in COLORS:
        s = series(impl)
        if s.empty:
            continue
        ax.errorbar(s.N, s.kernel_med * 1e3, yerr=errbars(s, scale=1e3), marker="o", capsize=3,
                    color=COLORS[impl], label=LABELS[impl])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Grid size N (N x N cells)"); ax.set_ylabel(f"Kernel-only runtime for K={K_main} steps (ms)")
    ax.set_title("Runtime vs grid size (median, IQR bars; CUDA = best block shape)\n" + sub_title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    P.finish(fig, "01_runtime_vs_size.png")

    # --- 2 speedup vs size
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sn, st = series("cuda_naive"), series("cuda_tiled_shared")
    if not sn.empty:
        ax.plot(sn.N, sn.spd_kernel_vs_serial, "-", marker="o", color="#6b6b6b", label="naive vs serial (kernel-only)")
        ax.plot(sn.N, sn.spd_kernel_vs_omp, "-", marker="s", color="#1f77b4", label="naive vs OpenMP (kernel-only)")
        ax.plot(sn.N, sn.spd_e2e_vs_omp, ":", marker="^", color="#d62728", label="naive vs OpenMP (END-TO-END, incl. PCIe)")
    if not st.empty:
        ax.plot(st.N, st.spd_kernel_vs_omp, "--", marker="d", color=COLORS["cuda_tiled_shared"], label="tiled vs OpenMP (kernel-only)")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Grid size N"); ax.set_ylabel("Speedup (x)")
    ax.set_title("GPU speedup vs grid size -- serial and OpenMP baselines shown separately\n" + sub_title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    P.finish(fig, "02_speedup_vs_size.png")

    # --- 3 block shapes
    blk = agg[(agg.impl.str.startswith("cuda")) & (agg.K == K_main)]
    Nb = [N for N in sorted(blk.N.unique()) if blk[(blk.N == N) & (blk.impl == "cuda_naive")].shape[0] >= 3]
    if Nb:
        fig, axes = plt.subplots(1, len(Nb), figsize=(5.2 * len(Nb), 4.8), squeeze=False)
        for ax, N in zip(axes[0], Nb):
            shapes = [s for s in SHAPE_ORDER if not blk[(blk.N == N) & (blk.block_x == s[0]) & (blk.block_y == s[1])].empty]
            x = np.arange(len(shapes)); w = 0.38
            for off, impl in ((-w / 2, "cuda_naive"), (w / 2, "cuda_tiled_shared")):
                rows = [blk[(blk.impl == impl) & (blk.N == N) & (blk.block_x == s[0]) & (blk.block_y == s[1])] for s in shapes]
                vals = [r.iloc[0] if len(r) else None for r in rows]
                y = [v.kernel_med * 1e3 if v is not None else np.nan for v in vals]
                e = [[(v.kernel_med - v.kernel_q1) * 1e3 if v is not None else 0 for v in vals],
                     [(v.kernel_q3 - v.kernel_med) * 1e3 if v is not None else 0 for v in vals]]
                ax.bar(x + off, y, w, yerr=e, capsize=3, color=COLORS[impl], label=LABELS[impl])
            ax.set_xticks(x); ax.set_xticklabels([f"{a}x{b}" for a, b in shapes])
            ax.set_title(f"N = {N}"); ax.set_xlabel("Block shape (x*y threads)"); ax.set_ylabel("Kernel time (ms)")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"CUDA block-shape sensitivity, K={K_main} (median, IQR)  --  " + gpu_name, fontsize=10)
        P.finish(fig, "03_block_shapes.png")

    # --- 4 naive vs tiled
    t = agg[(agg.impl == "cuda_tiled_shared") & (agg.K == K_main) & agg.opt_speedup_naive_over_tiled.notna()]
    if not t.empty:
        Nt = sorted(t.N.unique())
        shapes = [s for s in SHAPE_ORDER if not t[(t.block_x == s[0]) & (t.block_y == s[1])].empty]
        fig, ax = plt.subplots(figsize=(8, 5))
        w = 0.8 / len(Nt)
        for k, N in enumerate(Nt):
            vals = []
            for s in shapes:
                r = t[(t.N == N) & (t.block_x == s[0]) & (t.block_y == s[1])]
                vals.append(r.iloc[0].opt_speedup_naive_over_tiled if len(r) else np.nan)
            ax.bar(np.arange(len(shapes)) + k * w - 0.4 + w / 2, vals, w, label=f"N={N}")
        ax.axhline(1, color="k", lw=1)
        ax.set_xticks(np.arange(len(shapes))); ax.set_xticklabels([f"{a}x{b}" for a, b in shapes])
        ax.set_ylabel("Optimisation speedup  T_naive / T_tiled   (>1 = tiling helps)")
        ax.set_xlabel("Block shape"); ax.set_title("Does shared-memory tiling help?  " + gpu_name, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3); ax.legend()
        P.finish(fig, "04_naive_vs_tiled.png")

    # --- 5 steps sweep: launch + transfer overhead
    nv = agg[(agg.impl == "cuda_naive")]
    cand = nv.groupby(["N", "block_x", "block_y"]).K.nunique()
    cand = cand[cand >= 3]
    if len(cand):
        N0, bx, by = cand.index[0]
        g = nv[(nv.N == N0) & (nv.block_x == bx) & (nv.block_y == by)].sort_values("K")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(g.K, g.kernel_med * 1e3, "o-", color=COLORS["cuda_naive"], label=f"CUDA naive {bx}x{by} kernel-only")
        ax.plot(g.K, g.e2e_med * 1e3, "o--", color="#d62728", label="CUDA naive end-to-end (incl. PCIe copies)")
        for impl in ("cpu_serial", "cpu_omp"):
            c = agg[(agg.impl == impl) & (agg.N == N0)].sort_values("K")
            if len(c) > 1:
                ax.plot(c.K, c.kernel_med * 1e3, "s:", color=COLORS[impl], label=LABELS[impl])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of time steps K")
        ax.set_ylabel("Runtime (ms)"); ax.set_title(f"N = {N0}: runtime vs steps"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(g.K, g.transfer_pct_of_e2e, "o-", color="#d62728")
        ax.set_xscale("log"); ax.set_xlabel("Number of time steps K"); ax.set_ylabel("Host<->device copy share of end-to-end time (%)")
        ax.set_title("Transfer overhead is amortised as K grows"); ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(gpu_name, fontsize=9)
        P.finish(fig, "05_kernel_vs_end_to_end.png")

    # --- 6 bandwidth
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drew = False
    for impl in ("cuda_naive", "cuda_tiled_shared"):
        s = series(impl)
        if s.empty:
            continue
        drew = True
        ax.plot(s.N, s.pct_peak_8B, "o-", color=COLORS[impl], label=f"{LABELS[impl]} -- 8 B/update model")
        ax.plot(s.N, s.pct_peak_24B, "x:", color=COLORS[impl], alpha=0.7, label=f"{LABELS[impl]} -- 24 B/update model (not physical)")
    if drew:
        ax.axhline(100, color="red", lw=1.2); ax.text(Ns[0], 103, "100 % of theoretical peak DRAM bandwidth", color="red", fontsize=8)
        ax.set_xscale("log", base=2); ax.set_xlabel("Grid size N"); ax.set_ylabel("Estimated bandwidth, % of peak")
        ax.set_title(f"Estimated effective bandwidth vs theoretical peak ({peak:.0f} GB/s)\nestimates from a traffic model, NOT measured DRAM traffic", fontsize=9)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
        P.finish(fig, "06_bandwidth_pct_peak.png")
    else:
        plt.close(fig)

    # --- 7 OpenMP threads
    om = agg[(agg.impl == "cpu_omp") & (agg.K == K_main)]
    cnt = om.groupby("N").threads.nunique()
    cnt = cnt[cnt >= 2]
    if len(cnt):
        N0 = cnt.index.max()
        o = om[om.N == N0].sort_values("threads")
        s = agg[(agg.impl == "cpu_serial") & (agg.N == N0) & (agg.K == K_main)]
        if len(s):
            sp = s.iloc[0].kernel_med / o.kernel_med
            fig, ax = plt.subplots(figsize=(6.5, 5))
            ax.plot(o.threads, sp, "o-", color=COLORS["cpu_omp"], label="measured")
            ax.plot(o.threads, o.threads, "k--", label="ideal linear")
            ax.set_xlabel("OpenMP threads"); ax.set_ylabel("Speedup over serial")
            ax.set_title(f"CPU thread scaling, N={N0}  ({env.get('cpu_threads_available','?')} hardware threads available)", fontsize=9)
            ax.grid(True, alpha=0.3); ax.legend()
            P.finish(fig, "07_openmp_threads.png")

    # --- 8 throughput
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for impl in COLORS:
        s = series(impl)
        if not s.empty:
            ax.plot(s.N, s.throughput_Gcells_s, "o-", color=COLORS[impl], label=LABELS[impl])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("Grid size N")
    ax.set_ylabel("Throughput (Gcell-updates / s)"); ax.set_title("Cell-update throughput vs grid size\n" + sub_title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    P.finish(fig, "08_throughput_vs_size.png")


# ------------------------------------------------------------------ tables
def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{floatfmt}f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tables(agg, env, peak, K_main, synthetic):
    L = []
    if synthetic:
        L.append("> **WARNING: SYNTHETIC GPU DATA. This file was produced by the pipeline test (`--mock`). Do not use in the report.**\n")
    L.append("# Benchmark summary (auto-generated by benchmarks/analyze.py)\n")
    L.append(f"- GPU: **{env.get('gpu_name','?')}**  | theoretical peak bandwidth used: **{peak:.1f} GB/s** (from the CUDA harness)")
    L.append(f"- CPU: **{env.get('cpu_model','?')}**, {env.get('cpu_threads_available','?')} hardware threads available")
    L.append(f"- {env.get('nvcc','')} | CPU flags `{env.get('cpu_flags','')}` | nvcc flags `{env.get('nvcc_flags','')}`")
    L.append(f"- commit {env.get('git_commit','?')} | captured {env.get('captured_utc','?')}")
    L.append("- Times are medians over all repetitions and rounds. Kernel-only excludes PCIe transfers; end-to-end includes them (GPU only).\n")

    # T1 scaling
    rows = []
    for N in sorted(agg[agg.K == K_main].N.unique()):
        s = agg[(agg.impl == "cpu_serial") & (agg.N == N) & (agg.K == K_main)]
        n = best_shape(agg, "cuda_naive", N, K_main)
        t = best_shape(agg, "cuda_tiled_shared", N, K_main)
        o = agg[(agg.impl == "cpu_omp") & (agg.N == N) & (agg.K == K_main) & (agg.threads == env.get("cpu_threads_available", -1))]
        r = {"N": N,
             "serial ms": s.iloc[0].kernel_med * 1e3 if len(s) else np.nan,
             "OpenMP ms": o.iloc[0].kernel_med * 1e3 if len(o) else np.nan,
             "naive ms": n.kernel_med * 1e3 if n is not None else np.nan,
             "naive blk": f"{int(n.block_x)}x{int(n.block_y)}" if n is not None else "",
             "tiled ms": t.kernel_med * 1e3 if t is not None else np.nan,
             "naive x vs serial": n.spd_kernel_vs_serial if n is not None else np.nan,
             "naive x vs OMP": n.spd_kernel_vs_omp if n is not None else np.nan,
             "naive e2e x vs OMP": n.spd_e2e_vs_omp if n is not None else np.nan,
             "naive Gcell/s": n.throughput_Gcells_s if n is not None else np.nan,
             "naive %peak (8B)": n.pct_peak_8B if n is not None else np.nan}
        rows.append(r)
    if rows:
        L.append(f"## Table 1 - CPU vs GPU across grid sizes (K = {K_main}; CUDA rows use the fastest block shape for that N)\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

    # T2 blocks
    blk = agg[agg.impl.str.startswith("cuda") & (agg.K == K_main)]
    for N in sorted(blk.N.unique()):
        sub = blk[blk.N == N]
        if sub[sub.impl == "cuda_naive"].shape[0] < 3:
            continue
        rows = []
        for s in SHAPE_ORDER:
            a = sub[(sub.impl == "cuda_naive") & (sub.block_x == s[0]) & (sub.block_y == s[1])]
            b = sub[(sub.impl == "cuda_tiled_shared") & (sub.block_x == s[0]) & (sub.block_y == s[1])]
            if a.empty and b.empty:
                continue
            rows.append({"block": f"{s[0]}x{s[1]}",
                         "naive ms": a.iloc[0].kernel_med * 1e3 if len(a) else np.nan,
                         "naive IQR %": a.iloc[0].iqr_pct if len(a) else np.nan,
                         "tiled ms": b.iloc[0].kernel_med * 1e3 if len(b) else np.nan,
                         "tiled IQR %": b.iloc[0].iqr_pct if len(b) else np.nan,
                         "naive/tiled": b.iloc[0].opt_speedup_naive_over_tiled if len(b) else np.nan,
                         "naive %peak(8B)": a.iloc[0].pct_peak_8B if len(a) else np.nan})
        L.append(f"## Table 2 - block-shape experiment, N = {N}, K = {K_main}\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

    # T3 steps
    nv = agg[agg.impl == "cuda_naive"]
    cand = nv.groupby(["N", "block_x", "block_y"]).K.nunique()
    cand = cand[cand >= 3]
    if len(cand):
        N0, bx, by = cand.index[0]
        rows = []
        for K in sorted(nv[nv.N == N0].K.unique()):
            g = nv[(nv.N == N0) & (nv.K == K) & (nv.block_x == bx) & (nv.block_y == by)]
            if g.empty:
                continue
            g = g.iloc[0]
            rows.append({"K": K, "kernel ms": g.kernel_med * 1e3, "end-to-end ms": g.e2e_med * 1e3,
                         "transfer ms": g.transfer_ms, "transfer % of e2e": g.transfer_pct_of_e2e,
                         "per-step us": g.kernel_med / K * 1e6,
                         "e2e x vs serial": g.spd_e2e_vs_serial, "e2e x vs OMP": g.spd_e2e_vs_omp})
        L.append(f"## Table 3 - time-step sweep (launch / transfer overhead), N = {N0}, naive {bx}x{by}\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

    # T4 threads
    om = agg[(agg.impl == "cpu_omp") & (agg.K == K_main)]
    cnt = om.groupby("N").threads.nunique()
    cnt = cnt[cnt >= 2]
    if len(cnt):
        N0 = cnt.index.max()
        s = agg[(agg.impl == "cpu_serial") & (agg.N == N0) & (agg.K == K_main)]
        rows = []
        for _, o in om[om.N == N0].sort_values("threads").iterrows():
            sp = s.iloc[0].kernel_med / o.kernel_med if len(s) else np.nan
            rows.append({"threads": int(o.threads), "ms": o.kernel_med * 1e3, "speedup vs serial": sp,
                         "parallel efficiency %": 100 * sp / o.threads})
        L.append(f"## Table 4 - OpenMP thread scaling, N = {N0}, K = {K_main} (hardware threads available: {env.get('cpu_threads_available','?')})\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

    # T5 noise
    noisy = agg[(agg.iqr_pct > 8) | (agg.round_spread_pct > 8)].sort_values("round_spread_pct", ascending=False)
    L.append("## Table 5 - noisy configurations (IQR > 8 % or round-to-round range > 8 % of median)\n")
    if noisy.empty:
        L.append("None.\n")
    else:
        t = noisy[["impl", "N", "K", "block_x", "block_y", "threads", "n", "iqr_pct", "round_spread_pct"]].head(25)
        L.append(md(t) + "\n")
        L.append("Report these with their spread, repeat them, or exclude them with an explanation. Do not silently drop them.\n")

    # fact checks
    L.append("## Automatic fact checks (data only - YOU still write the explanation)\n")
    facts = []
    t = agg[(agg.impl == "cuda_tiled_shared") & agg.opt_speedup_naive_over_tiled.notna()]
    if len(t):
        helped = (t.opt_speedup_naive_over_tiled > 1.02).sum()
        hurt = (t.opt_speedup_naive_over_tiled < 0.98).sum()
        facts.append(f"Tiling vs naive over {len(t)} matched configs: faster (>2 %) in {helped}, slower (>2 %) in {hurt}, "
                     f"within +/-2 % in {len(t)-helped-hurt}. Range {t.opt_speedup_naive_over_tiled.min():.2f}x - {t.opt_speedup_naive_over_tiled.max():.2f}x.")
    g = agg[(agg.impl == "cuda_naive") & (agg.K == K_main)]
    if len(g):
        top = g.sort_values("kernel_med").drop_duplicates("N").sort_values("N")
        lo, hi = top.iloc[0], top.iloc[-1]
        facts.append(f"Naive CUDA kernel-only speedup vs serial: {lo.spd_kernel_vs_serial:.1f}x at N={int(lo.N)} -> {hi.spd_kernel_vs_serial:.1f}x at N={int(hi.N)}; "
                     f"vs OpenMP: {lo.spd_kernel_vs_omp:.1f}x -> {hi.spd_kernel_vs_omp:.1f}x.")
        win = top[top.spd_kernel_vs_omp > 1]
        facts.append("GPU beats the OpenMP baseline (kernel-only) at every tested size." if len(win) == len(top) else
                     f"GPU does NOT beat OpenMP (kernel-only) at N = {', '.join(str(int(n)) for n in top[top.spd_kernel_vs_omp <= 1].N)}.")
        win2 = top[top.spd_e2e_vs_omp > 1]
        if len(win2) != len(top):
            facts.append(f"END-TO-END the GPU does NOT beat OpenMP at N = {', '.join(str(int(n)) for n in top[top.spd_e2e_vs_omp <= 1].N)}.")
        if not np.isnan(peak):
            facts.append(f"Naive best-shape bandwidth, 8 B/update model: {top.pct_peak_8B.min():.0f} % - {top.pct_peak_8B.max():.0f} % of the {peak:.0f} GB/s peak "
                         f"(24 B model: {top.pct_peak_24B.min():.0f} % - {top.pct_peak_24B.max():.0f} %). A value above 100 % means the model over-counts DRAM traffic (cache reuse).")
    om = agg[(agg.impl == "cpu_omp") & (agg.K == K_main) & (agg.threads == env.get("cpu_threads_available", -1)) & agg.spd_kernel_vs_serial.notna()]
    if len(om):
        facts.append(f"OpenMP ({env.get('cpu_threads_available','?')} threads) vs serial: {om.spd_kernel_vs_serial.min():.2f}x - {om.spd_kernel_vs_serial.max():.2f}x over the grid sizes.")
    L.extend("- " + f for f in facts)
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--K", type=int, default=1000, help="main time-step count used for the size/block plots")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    raw, env = load(out)
    agg = aggregate(raw)
    agg, peak = add_metrics(agg, env, raw)
    synthetic = agg["device"].astype(str).str.contains("MOCK|SYNTHETIC", regex=True).any()
    if synthetic:
        print("*** SYNTHETIC GPU DATA DETECTED - outputs are watermarked, do not use in the report ***")
    if env.get("gpu_name", "none detected") == "none detected":
        dev = agg[agg.impl.str.startswith("cuda")]["device"]
        if len(dev):
            env["gpu_name"] = str(dev.iloc[0])
    agg.to_csv(out / "summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    (out / "summary_tables.md").write_text(tables(agg, env, peak, a.K, synthetic))
    print(f"wrote {out/'summary.csv'} and {out/'summary_tables.md'}")
    plots(agg, env, peak, a.K, Plotter(out / "plots", synthetic))


if __name__ == "__main__":
    main()
