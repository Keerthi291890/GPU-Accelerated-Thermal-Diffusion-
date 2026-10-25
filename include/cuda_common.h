/*
 * cuda_common.h  --  shared host-side harness for the CUDA heat-diffusion programs
 * (Person 2).  Re-uses Person 1's include/simulation_config.h for the default
 * parameters, and reproduces EXACTLY the same initial grid, boundary treatment
 * and CSV format as src/cpu_serial.c so results can be compared cell by cell.
 *
 * Model (identical to the CPU code):
 *   next[i][j] = c + r * (up + down + left + right - 4*c)     (interior cells only)
 *   boundary cells stay at AMBIENT_TEMP (fixed / Dirichlet)
 *   row-major layout: idx = row * N + col   (row = y = i,  col = x = j)
 */
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "simulation_config.h"   /* GRID_SIZE, TIME_STEPS, AMBIENT_TEMP, HOT_TEMP, HOTSPOT_RADIUS, DIFFUSION_R */

/* ---------- error checking: wrap EVERY CUDA call ---------- */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ---------- same initial condition as cpu_serial.c ---------- */
static void initialize_grid(float *grid, int N)
{
    for (int i = 0; i < N * N; i++) grid[i] = AMBIENT_TEMP;
    int center = N / 2;
    for (int i = center - HOTSPOT_RADIUS; i < center + HOTSPOT_RADIUS; i++)
        for (int j = center - HOTSPOT_RADIUS; j < center + HOTSPOT_RADIUS; j++)
            grid[i * N + j] = HOT_TEMP;
}

/* same format as cpu_serial.c: comma separated, no header, %.9g */
static void save_csv(const char *filename, const float *grid, int N)
{
    FILE *f = fopen(filename, "w");
    if (!f) { perror(filename); return; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j > 0) fputc(',', f);
            fprintf(f, "%.9g", grid[i * N + j]);
        }
        fputc('\n', f);
    }
    fclose(f);
}

/* ---------- GPU information (for the architecture / cloud-HPC report) ---------- */
static void print_device_info(void)
{
    int dev = 0, rt = 0, drv = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));

    int clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz,     cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits,      cudaDevAttrGlobalMemoryBusWidth, dev);
    /* DDR/GDDR: 2 transfers per clock */
    double peak_bw = 2.0 * (double)mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;

    printf("---------------- GPU device information ----------------\n");
    printf("Device                : %s\n", p.name);
    printf("Compute capability    : %d.%d\n", p.major, p.minor);
    printf("CUDA runtime / driver : %d.%d / %d.%d\n", rt / 1000, (rt % 100) / 10, drv / 1000, (drv % 100) / 10);
    printf("Streaming MPs (SMs)   : %d\n", p.multiProcessorCount);
    printf("Warp size             : %d\n", p.warpSize);
    printf("Max threads / block   : %d\n", p.maxThreadsPerBlock);
    printf("Max threads / SM      : %d\n", p.maxThreadsPerMultiProcessor);
    printf("Shared mem / block    : %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("L2 cache              : %d KB\n", p.l2CacheSize / 1024);
    printf("Global memory         : %.1f GB\n", p.totalGlobalMem / 1073741824.0);
    printf("Core clock            : %.0f MHz\n", clock_khz / 1000.0);
    printf("Memory bus / clock    : %d-bit / %.0f MHz\n", bus_bits, mem_clock_khz / 1000.0);
    printf("Theoretical peak BW   : %.1f GB/s\n", peak_bw);
    printf("--------------------------------------------------------\n");
}

/* ---------- the harness ---------- */
typedef void (*StepLauncher)(const float *d_in, float *d_out, int N, float r, int bx, int by);

struct Options {
    int N = GRID_SIZE;
    int K = TIME_STEPS;
    int bx = 16, by = 16;
    int reps = 5;
    const char *save_path = nullptr;                         /* final grid CSV (for validation) */
    const char *csv_path  = "results/cuda/runtime_cuda.csv"; /* timing rows (for Person 3)      */
};

/* usage: prog [N] [K] [block_x] [block_y] [reps] [final_grid.csv] */
static Options parse_args(int argc, char **argv)
{
    Options o;
    if (argc > 1) o.N    = atoi(argv[1]);
    if (argc > 2) o.K    = atoi(argv[2]);
    if (argc > 3) o.bx   = atoi(argv[3]);
    if (argc > 4) o.by   = atoi(argv[4]);
    if (argc > 5) o.reps = atoi(argv[5]);
    if (argc > 6) o.save_path = argv[6];
    if (o.N < 3 || o.K < 1 || o.bx < 1 || o.by < 1 || o.reps < 1 ||
        o.bx * o.by > 1024 || DIFFUSION_R <= 0.0f || DIFFUSION_R > 0.25f) {
        fprintf(stderr, "usage: %s [N=%d] [K=%d] [block_x=16] [block_y=16] [reps=5] [final_grid.csv]\n"
                        "       need N>=3, K>=1, block_x*block_y<=1024, 0<r<=0.25\n",
                argv[0], GRID_SIZE, TIME_STEPS);
        exit(EXIT_FAILURE);
    }
    return o;
}

static double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int run_experiment(const char *name, StepLauncher launch, int argc, char **argv)
{
    Options o = parse_args(argc, argv);
    const int N = o.N, K = o.K;
    const size_t cells = (size_t)N * N;
    const size_t bytes = cells * sizeof(float);
    const float r = DIFFUSION_R;

    print_device_info();
    printf("Implementation : %s\n", name);
    printf("Grid           : %d x %d   Timesteps: %d   r = %.2f\n", N, N, K, r);
    printf("Block shape    : %d x %d (%d threads, %d warps)\n", o.bx, o.by, o.bx * o.by, (o.bx * o.by + 31) / 32);

    /* host data: initial grid, identical to the CPU reference */
    std::vector<float> h_init(cells), h_out(cells);
    initialize_grid(h_init.data(), N);

    /* device data: two buffers (double buffering).  BOTH start as the initial grid
     * so the fixed boundary cells are already correct in each buffer; the kernels
     * only ever write interior cells.                                            */
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    cudaEvent_t e_start, e_k0, e_k1, e_end;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_k0));
    CUDA_CHECK(cudaEventCreate(&e_k1));
    CUDA_CHECK(cudaEventCreate(&e_end));

    /* ---- warm-up (not timed): context creation, clock ramp-up, caches ---- */
    CUDA_CHECK(cudaMemcpy(d_a, h_init.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_init.data(), bytes, cudaMemcpyHostToDevice));
    {
        float *in = d_a, *out = d_b;
        for (int s = 0; s < 10; s++) { launch(in, out, N, r, o.bx, o.by); std::swap(in, out); }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> t_kernel, t_total;

    for (int rep = 0; rep < o.reps; rep++) {
        CUDA_CHECK(cudaEventRecord(e_start));
        CUDA_CHECK(cudaMemcpy(d_a, h_init.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_init.data(), bytes, cudaMemcpyHostToDevice));

        float *in = d_a, *out = d_b;
        CUDA_CHECK(cudaEventRecord(e_k0));
        for (int s = 0; s < K; s++) {
            launch(in, out, N, r, o.bx, o.by);
            std::swap(in, out);                 /* new -> old, exactly like the CPU pointer swap */
        }
        CUDA_CHECK(cudaEventRecord(e_k1));
        CUDA_CHECK(cudaGetLastError());         /* catches launch-configuration errors */

        /* after the final swap, 'in' points at the newest grid */
        CUDA_CHECK(cudaMemcpy(h_out.data(), in, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e_end));
        CUDA_CHECK(cudaEventSynchronize(e_end));

        float ms_kernel = 0.f, ms_total = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, e_k0, e_k1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, e_start, e_end));
        t_kernel.push_back(ms_kernel / 1e3);
        t_total.push_back(ms_total / 1e3);
        printf("  run %2d: kernel-only %.6f s   end-to-end %.6f s\n", rep + 1, t_kernel.back(), t_total.back());
    }

    double k_med = median_of(t_kernel);
    double k_min = *std::min_element(t_kernel.begin(), t_kernel.end());
    double k_max = *std::max_element(t_kernel.begin(), t_kernel.end());
    double e_med = median_of(t_total);

    double updates = (double)(N - 2) * (N - 2) * (double)K;       /* interior cell updates */
    double throughput = updates / k_med;                          /* cell updates / s      */
    double bw = 24.0 * updates / k_med / 1e9;                     /* GB/s: 5 reads + 1 write, float32 */

    int c = N / 2;
    printf("Median kernel-only : %.6f s   (min %.6f, max %.6f)\n", k_med, k_min, k_max);
    printf("Median end-to-end  : %.6f s   (includes host<->device copies)\n", e_med);
    printf("Throughput         : %.1f Mcell-updates/s\n", throughput / 1e6);
    printf("Effective bandwidth: %.1f GB/s  (24 B/update model, NOT measured DRAM traffic)\n", bw);
    printf("Center temperature : %.6f   (CPU reference for N=1024,K=1000: 71.636833)\n", h_out[(size_t)c * N + c]);

    if (o.save_path) {
        save_csv(o.save_path, h_out.data(), N);
        printf("Final grid saved to %s\n", o.save_path);
    }

    /* append one summary row (same spirit as Person 1's runtime_*.csv) for Person 3 */
    int rc = system("mkdir -p results/cuda");
    (void)rc;
    FILE *chk = fopen(o.csv_path, "r");
    bool need_header = (chk == nullptr);
    if (chk) fclose(chk);
    FILE *csv = fopen(o.csv_path, "a");
    if (csv) {
        if (need_header)
            fprintf(csv, "implementation,grid_size,time_steps,block_x,block_y,reps,"
                         "kernel_median_s,kernel_min_s,kernel_max_s,end_to_end_median_s,"
                         "throughput_Mcells_per_s,effective_bw_GBps\n");
        fprintf(csv, "%s,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.3f,%.3f\n",
                name, N, K, o.bx, o.by, o.reps, k_med, k_min, k_max, e_med, throughput / 1e6, bw);
        fclose(csv);
        printf("Timing row appended to %s\n", o.csv_path);
    }

    CUDA_CHECK(cudaEventDestroy(e_start));
    CUDA_CHECK(cudaEventDestroy(e_k0));
    CUDA_CHECK(cudaEventDestroy(e_k1));
    CUDA_CHECK(cudaEventDestroy(e_end));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return EXIT_SUCCESS;
}

#endif /* CUDA_COMMON_H */
