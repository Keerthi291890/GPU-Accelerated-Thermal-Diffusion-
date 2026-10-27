/*
 * cpu_bench.c -- benchmark-grade CPU implementations (serial + OpenMP) with RUNTIME
 * parameters, so the grid-size / thread-count sweeps need no recompilation.
 *
 * Why this file exists (Person 3):
 *   src/cpu_serial.c and src/cpu_openmp.c fix GRID_SIZE / TIME_STEPS at compile time and
 *   always write 3.5 MB+ CSV files, which makes sweeps awkward. This program uses
 *   EXACTLY the same maths, initial condition and fixed (Dirichlet) boundary as Person 1's
 *   code; its output is validated cell-for-cell against results/cpu/final_serial.csv
 *   (see tests/test_correctness.py and benchmarks/validate_all.py).
 *
 *   One deliberate difference: both buffers start as a copy of the initial grid, so the
 *   boundary cells are never touched again. Person 1's serial code instead re-fills the
 *   WHOLE next grid with ambient every step (an extra full-grid memory pass that the
 *   OpenMP and CUDA versions do not do). Removing it makes serial / OpenMP / CUDA do
 *   the same amount of work. The numerical result is identical.
 *
 * Build:  gcc -O3 -ffp-contract=off -fopenmp benchmarks/cpu_bench.c -o cpu_bench
 *         (-ffp-contract=off: no fused multiply-add, same arithmetic as the CUDA --fmad=false build)
 * Usage:  ./cpu_bench MODE N K THREADS REPS [final_grid.csv]
 *           MODE    serial | omp          (serial = plain loops, no OpenMP runtime involved)
 *           THREADS number of OpenMP threads (0 = all available); ignored by serial
 *           REPS    timed repetitions (each starts from the initial grid)
 * Output: human-readable lines plus machine-parsable lines starting with "RESULT ".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define AMBIENT_TEMP    25.0f
#define HOT_TEMP        100.0f
#define HOTSPOT_RADIUS  25
#define DIFFUSION_R     0.2f     /* r = alpha*dt/dx^2, must be <= 0.25 (2-D explicit stability) */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void init_grid(float *g, int N)
{
    for (size_t i = 0; i < (size_t)N * N; i++) g[i] = AMBIENT_TEMP;
    int c = N / 2;
    for (int i = c - HOTSPOT_RADIUS; i < c + HOTSPOT_RADIUS; i++)
        for (int j = c - HOTSPOT_RADIUS; j < c + HOTSPOT_RADIUS; j++)
            g[(size_t)i * N + j] = HOT_TEMP;
}

static void save_csv(const char *path, const float *g, int N)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j) fputc(',', f);
            fprintf(f, "%.9g", g[(size_t)i * N + j]);
        }
        fputc('\n', f);
    }
    fclose(f);
}

/* returns pointer to the buffer holding the newest grid */
static float *run_serial(float *a, float *b, int N, int K)
{
    float *in = a, *out = b;
    for (int s = 0; s < K; s++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                size_t idx = (size_t)i * N + j;
                float c = in[idx], up = in[idx - N], dn = in[idx + N], lf = in[idx - 1], rt = in[idx + 1];
                out[idx] = c + DIFFUSION_R * (up + dn + lf + rt - 4.0f * c);
            }
        }
        float *t = in; in = out; out = t;
    }
    return in;
}

static float *run_omp(float *a, float *b, int N, int K)
{
    float *in = a, *out = b;
    /* one persistent parallel region (no fork/join per step); the implicit barriers of
       'omp for' and 'omp single' separate consecutive time steps */
    #pragma omp parallel shared(in, out)
    {
        for (int s = 0; s < K; s++) {
            #pragma omp for schedule(static)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    size_t idx = (size_t)i * N + j;
                    float c = in[idx], up = in[idx - N], dn = in[idx + N], lf = in[idx - 1], rt = in[idx + 1];
                    out[idx] = c + DIFFUSION_R * (up + dn + lf + rt - 4.0f * c);
                }
            }
            #pragma omp single
            { float *t = in; in = out; out = t; }
        }
    }
    return in;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s serial|omp N K THREADS REPS [final_grid.csv]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mode = argv[1];
    int N = atoi(argv[2]), K = atoi(argv[3]), threads = atoi(argv[4]), reps = atoi(argv[5]);
    const char *save_path = (argc > 6) ? argv[6] : NULL;
    int is_omp = (strcmp(mode, "omp") == 0);
    if (!is_omp && strcmp(mode, "serial") != 0) { fprintf(stderr, "MODE must be serial or omp\n"); return EXIT_FAILURE; }
    if (N < 2 * HOTSPOT_RADIUS + 2 || K < 1 || reps < 1 || DIFFUSION_R <= 0.0f || DIFFUSION_R > 0.25f) {
        /* the 50x50 hotspot is written at [N/2-25, N/2+25) -> smaller grids would write out of bounds */
        fprintf(stderr, "need N>=%d (hotspot must fit), K>=1, REPS>=1 and 0<r<=0.25\n", 2 * HOTSPOT_RADIUS + 2);
        return EXIT_FAILURE;
    }
    if (is_omp) { if (threads > 0) omp_set_num_threads(threads); threads = omp_get_max_threads(); }
    else threads = 1;

    size_t cells = (size_t)N * N, bytes = cells * sizeof(float);
    float *init = malloc(bytes), *a = malloc(bytes), *b = malloc(bytes);
    if (!init || !a || !b) { fprintf(stderr, "allocation failed (%zu MB x3)\n", bytes >> 20); return EXIT_FAILURE; }
    init_grid(init, N);

    printf("Implementation : cpu_%s   Grid: %d x %d   Steps: %d   Threads: %d   (omp procs available: %d)\n",
           mode, N, N, K, threads, omp_get_num_procs());

    /* warm-up (untimed): page-faults the buffers in, wakes the OpenMP thread pool */
    memcpy(a, init, bytes); memcpy(b, init, bytes);
    (is_omp ? run_omp : run_serial)(a, b, N, K < 10 ? K : 10);

    float *result = NULL;
    for (int rep = 0; rep < reps; rep++) {
        memcpy(a, init, bytes); memcpy(b, init, bytes);          /* untimed reset */
        double t0 = now_s();
        result = (is_omp ? run_omp : run_serial)(a, b, N, K);
        double t1 = now_s();
        printf("RESULT cpu_%s %d %d %d %d %.9f\n", mode, N, K, threads, rep + 1, t1 - t0);
    }

    double sum = 0.0;
    for (size_t i = 0; i < cells; i++) sum += result[i];
    printf("Center temperature : %.6f   (reference for N=1024,K=1000: 71.636833)\n", result[(size_t)(N / 2) * N + N / 2]);
    printf("Grid sum           : %.6f\n", sum);
    if (save_path) { save_csv(save_path, result, N); printf("Final grid saved to %s\n", save_path); }

    free(init); free(a); free(b);
    return EXIT_SUCCESS;
}
