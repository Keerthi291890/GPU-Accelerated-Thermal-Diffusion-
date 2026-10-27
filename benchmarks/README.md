# benchmarks/ -- Person 3

| File | What it does |
|---|---|
| `cpu_bench.c` | Serial + OpenMP CPU simulation with **runtime** N / K / threads / reps (no recompiling per grid size). Same maths as Person 1's code; output verified bit-identical. |
| `run_benchmarks.py` | Builds everything, captures the environment, runs all experiments (scaling, blocks, steps, threads) in shuffled rounds, appends one row per repetition to `results/raw_timings.csv`. Resumable. |
| `analyze.py` | Reads the raw CSV, computes medians / IQR / speedups / throughput / bandwidth, writes `results/summary.csv`, `results/summary_tables.md` and 8 plots in `results/plots/`. |
| `validate_all.py` | Full correctness matrix (non-multiple grid sizes, 1 step vs many, every block shape, float32 vs float64, Person 1 cross-check) -> `results/correctness_full.csv`. |
| `reference.py` | Independent NumPy implementation used by the tests and the validator. |

Quick commands (repository root):

```bash
python -m pytest tests -q                        # CPU tests anywhere; CUDA tests only with a GPU
python benchmarks/validate_all.py                # correctness matrix (needs GPU for the CUDA rows)
python benchmarks/run_benchmarks.py --env-only   # just record the environment
python benchmarks/run_benchmarks.py --dry-run    # list what would run
python benchmarks/run_benchmarks.py              # full study (resumable; ~20-40 min on a T4)
python benchmarks/analyze.py                     # tables + plots
```

Pipeline test without a GPU (SYNTHETIC GPU numbers, watermarked, written to `results/mock/`, never use in the report):

```bash
python benchmarks/run_benchmarks.py --mock --out results/mock --sizes 128,256,512,1024 --rounds 2
python benchmarks/analyze.py --out results/mock
```
