/*
 * cuda_naive.cu  --  naive CUDA kernel: ONE THREAD = ONE GRID CELL.
 * Every thread reads its centre cell and its four neighbours straight from
 * global memory and writes one result.  No shared memory, no tricks.
 *
 * Build (Colab / Linux):
 *   nvcc -O3 --fmad=false -I include src/cuda_naive.cu -o cuda_naive
 * Run:
 *   ./cuda_naive [N] [K] [block_x] [block_y] [reps] [final_grid.csv]
 *   ./cuda_naive 1024 1000 16 16 5 results/cuda/final_cuda_naive.csv
 *
 * --fmad=false stops nvcc fusing a*b+c into a single FMA instruction, so the GPU
 * performs the same separate multiply and add as the CPU code and the results can
 * match the CPU reference (almost) bit-for-bit.
 */
#include "cuda_common.h"

__global__ void heat_naive(const float *__restrict__ in, float *__restrict__ out, int N, float r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   /* column (j) */
    int y = blockIdx.y * blockDim.y + threadIdx.y;   /* row    (i) */

    /* interior cells only; boundary cells are never written (they stay at ambient) */
    if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1) {
        int idx = y * N + x;
        float center = in[idx];
        float up     = in[idx - N];
        float down   = in[idx + N];
        float left   = in[idx - 1];
        float right  = in[idx + 1];
        /* same operation order as cpu_serial.c */
        out[idx] = center + r * (up + down + left + right - 4.0f * center);
    }
}

static void launch_naive(const float *d_in, float *d_out, int N, float r, int bx, int by)
{
    dim3 block(bx, by);
    dim3 grid((N + bx - 1) / bx, (N + by - 1) / by);   /* round up so the whole plate is covered */
    heat_naive<<<grid, block>>>(d_in, d_out, N, r);
}

int main(int argc, char **argv)
{
    return run_experiment("cuda_naive", launch_naive, argc, argv);
}
