# Member 3 additions - what to do with these files

Unzip at the **repository root**. Nothing from Members 1 or 2 is overwritten (new files only):

```
benchmarks/  cpu_bench.c  run_benchmarks.py  analyze.py  validate_all.py  reference.py  README.md
tests/       test_correctness.py  mock_cuda.py
docs/        benchmark_methodology.md  cloud_hpc_analysis.md
notebooks/   thermal_hpc_case_study.ipynb
```

Append to `.gitignore`:

```
build/
results/mock/
results/smoke/
results/_cuda_scratch/
*.exe
```

Then `git rm --cached cpu_openmp.exe cpu_serial.exe test.exe test.c` (Windows binaries; useless on Colab).
Commit and push, then open `notebooks/thermal_hpc_case_study.ipynb` in Colab (GPU runtime) and run it top to bottom.

## Verified here (CPU only)
* `cpu_bench` output is bit-identical to Person 1's `results/cpu/final_serial.csv` (0 of 1,048,576 cells differ).
* OpenMP == serial exactly; C code matches an independent NumPy implementation; boundaries fixed; maximum principle holds.
* Orchestrator, resume-after-interruption, analysis tables and all 8 plots work (tested with a clearly watermarked SYNTHETIC GPU).
* float32 vs float64 error after 1000 steps: 6.0e-5 degC (tolerance is 1e-4).

## NOT verified (no GPU in the sandbox) - first thing to check on Colab
* Compiling and running the CUDA programs through `run_benchmarks.py`; parsing of their stdout.
* The 30 CUDA tests in `tests/test_correctness.py` and the CUDA rows of `validate_all.py` (they skip without a GPU).
* Every GPU number. None exist yet except Member 2's single-size results.

## Fill in after the real run
`docs/cloud_hpc_analysis.md` has **[FILL]** markers (GPU model, CPU threads, noise, copy-share percentages, run minutes).
