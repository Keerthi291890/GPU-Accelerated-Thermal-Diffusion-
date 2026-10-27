# Cloud HPC Analysis (Person 3)

> Prices below were collected on **27 Oct 2026** from third-party aggregators (links at the end), not from the
> providers' own price pages. Re-check the official pages before submitting. Items marked **[FILL]** must be completed
> from your own measured results (`results/env/environment.json`, `results/summary_tables.md`).

## 1. What "cloud HPC" means in this project

The study never needed to buy or own a GPU. A rented, on-demand NVIDIA T4 supplied the CUDA hardware, and the whole
experiment is reproducible from a public repository plus a notebook. The cloud properties that matter for the
analysis are: **on-demand access**, **pay-per-use billing**, **elasticity** (a bigger GPU is one setting away),
and **no control over the physical machine**.

Colab is a hosted notebook service on shared virtual machines. It is convenient and cheap, but it is not a
dedicated HPC cluster: there is no job scheduler, no guaranteed hardware, and no multi-node scaling. This analysis
therefore treats it honestly as **entry-level cloud GPU access** and compares it with a dedicated cloud instance.

## 2. Environment actually used

| Item | Value |
|---|---|
| GPU | **[FILL: environment.json -> gpu_name]**, theoretical peak bandwidth **[FILL]** GB/s, SM count **[FILL]** |
| CPU | **[FILL: cpu_model]**, **[FILL]** hardware threads |
| CUDA toolkit / driver | **[FILL: nvcc]** |
| Session date | **[FILL: captured_utc]** |
| Run-to-run noise observed | **[FILL: Table 5 of summary_tables.md - largest IQR % and round-to-round range %]** |

## 3. Cost comparison

| Option | Hardware | Price | Source |
|---|---|---|---|
| Colab (pay-as-you-go / Pro) | shared T4, no guarantee | $0.10 per compute unit ($9.99 per 100); a T4 consumed roughly 1.2-2.0 units/hour in the cited measurements, i.e. about **$0.12-0.20 per hour** | [1][2] |
| AWS EC2 `g4dn.xlarge` on-demand | 1x T4, 4 vCPU, 16 GiB, dedicated to you | **$0.526/h** in us-east-1 (about $0.58-0.89/h in the other regions listed) | [3] |
| AWS `g4dn.xlarge` spot | same instance, interruptible | about **$0.23/h** in us-east-1 (varies a lot by region and hour) | [3] |
| Buying a GPU | consumer/workstation card | roughly $500-$3,000+ upfront (a general range quoted by [4]) | [4] |

**Cost of this study.** Estimate the GPU time of the full benchmark (**[FILL] minutes**, see timestamps in
`raw_timings.csv`). At the on-demand rate this is `minutes/60 x $0.526`; at the Colab rate roughly a quarter of that.
The point for the viva: the entire experiment costs well under a dollar in either case.

**Rent vs. buy.** Break-even hours = purchase price / hourly rental price. Using the lowest purchase figure above
($500) and the on-demand rate, that is about 500 / 0.526 = **950 GPU-hours**; at the Colab rate it is several
thousand hours. A student project runs for tens of hours, so renting wins unless the GPU would be busy most of the
year. (This ignores electricity, the host computer, depreciation and resale value.)

## 4. What the cloud setting changed in our results

* **Variable resources.** Shared hardware means timings drift between sessions; this is why measurements are
  interleaved, repeated, summarised by the median with its IQR, and taken in a single session.
* **Few CPU cores.** The VM's small CPU limits the OpenMP baseline (**[FILL]** x measured speedup on **[FILL]** threads,
  Table 4). A dedicated 16-vCPU instance such as `g4dn.4xlarge` would give OpenMP more room and *reduce* the GPU's
  relative advantage. State this when quoting the headline speedup.
* **PCIe transfers.** In the cloud the GPU is a separate device behind PCIe, so host-device copies are a real cost.
  The step sweep (Table 3) shows the copy share falling from **[FILL]** % at K=10 to **[FILL]** % at K=10000: GPUs pay off
  when data stays on the device across many steps.
* **No profiler access.** Nsight Compute is awkward on Colab, so memory-bound behaviour is inferred from a bandwidth
  estimate (percentage of the T4's peak) instead of measured with hardware counters.
* **Session limits.** The free tier caps session length and idle time, and a session can be reassigned to different
  hardware. The benchmark runner is resumable (`raw_timings.csv` is appended after every job) for this reason.

## 5. Elasticity: what scaling in the cloud would look like

Not measured here, so do not claim results. Possible extensions, each a configuration change rather than a purchase:
a larger GPU for bigger grids (the 4096^2 grid uses ~134 MB for two float32 buffers, far below a T4's 16 GB, so memory
capacity is not the constraint at these sizes); several independent instances for a parameter sweep over r or
hotspot position (embarrassingly parallel across instances); multi-GPU domain decomposition, which would need halo
exchange between devices and is a different study.

## 6. Reproducing on a dedicated cloud instance (optional, recommended once)

Running the same notebook on one `g4dn.xlarge` for under an hour gives a second data point that makes the Colab
comparison much stronger: launch an instance with an NVIDIA/Deep Learning AMI, `git clone` the repository, run the
cells of `notebooks/thermal_hpc_case_study.ipynb` as shell commands, then **stop the instance**. Record its
`environment.json` next to the Colab one and compare noise (IQR) and speedups.

## 7. Likely viva questions

1. *Why is Colab not "real" HPC?* No scheduler, no multi-node interconnect, hardware not guaranteed, shared tenancy.
2. *Why median and not mean?* Shared-hardware noise produces outliers; the median is robust and the IQR shows spread.
3. *Is the speedup the same on another cloud?* No: it depends on the GPU, on the CPU baseline's core count, and on whether transfers are included.
4. *When is renting not better than buying?* When utilisation is high for a long period, or data cannot leave the premises.

## Sources (accessed 27 Oct 2026)

1. gpuperhour.com, "Google Colab Alternatives and What a GPU Costs by the Hour" (Colab Pro compute-unit costs).
2. Thunder Compute, "Google Colab alternatives" (compute-unit pricing; T4 about 1.76 units/h).
3. DoiT, AWS EC2 g4dn pricing pages (https://www.doit.com/compute/spot/us-east-1/g4dn.xlarge and other regions).
4. DEV Community, "Google Colab CLI review 2026" (rough local-GPU hardware range).
