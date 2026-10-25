/*
 * cuda_optimized.cu  --  shared-memory TILED kernel.
 * A thread block cooperatively copies a (BY+2) x (BX+2) tile (its own BX x BY cells
 * plus a 1-cell HALO on every side) from slow global memory into fast on-chip shared
 * memory, synchronises, then every thread computes from the tile.  Each global value
 * is loaded about once instead of being re-read by up to five different threads.
 *
 * Build:  nvcc -O3 --fmad=false -I include src/cuda_optimized.cu -o cuda_optimized
 * Run:    ./cuda_optimized [N] [K] [block_x] [block_y] [reps] [final_grid.csv]
 * Supported block shapes: 8x8, 16x16, 32x8, 32x16, 32x32 (the tile size is a compile-time
 * template parameter because shared-memory arrays need constant sizes).
 *
 * NOTE: shared memory is NOT guaranteed to win.  Modern GPUs have L1/L2 caches that
 * already catch most of the neighbour reuse, and the tile load + __syncthreads() add
 * overhead.  A measured slowdown is a valid result -- explain it in the report.
 */
#include "cuda_common.h"

template <int BX, int BY>
__global__ void heat_tiled(const float *__restrict__ in, float *__restrict__ out, int N, float r)
{
    __shared__ float tile[BY + 2][BX + 2];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * BX + tx;      /* global column */
    const int y = blockIdx.y * BY + ty;      /* global row    */
    const int sx = tx + 1, sy = ty + 1;      /* position inside the tile (shifted by the halo) */

    /* ---- phase 1: cooperative load. NO early return before __syncthreads(). ---- */
    if (x < N && y < N) tile[sy][sx] = in[y * N + x];                               /* own cell    */
    if (tx == 0      && x > 0     && y < N) tile[sy][0]      = in[y * N + x - 1];   /* left halo   */
    if (tx == BX - 1 && x + 1 < N && y < N) tile[sy][BX + 1] = in[y * N + x + 1];   /* right halo  */
    if (ty == 0      && y > 0     && x < N) tile[0][sx]      = in[(y - 1) * N + x]; /* top halo    */
    if (ty == BY - 1 && y + 1 < N && x < N) tile[BY + 1][sx] = in[(y + 1) * N + x]; /* bottom halo */
    /* (corner halo cells are never needed: the 5-point stencil has no diagonal neighbours) */

    __syncthreads();   /* the whole tile must be loaded before anyone reads it */

    /* ---- phase 2: compute from shared memory ---- */
    if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1) {
        float center = tile[sy][sx];
        float up     = tile[sy - 1][sx];
        float down   = tile[sy + 1][sx];
        float left   = tile[sy][sx - 1];
        float right  = tile[sy][sx + 1];
        out[y * N + x] = center + r * (up + down + left + right - 4.0f * center);
    }
}

template <int BX, int BY>
static void launch_shape(const float *d_in, float *d_out, int N, float r)
{
    dim3 block(BX, BY);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY);
    heat_tiled<BX, BY><<<grid, block>>>(d_in, d_out, N, r);
}

static void launch_tiled(const float *d_in, float *d_out, int N, float r, int bx, int by)
{
    if      (bx == 16 && by == 16) launch_shape<16, 16>(d_in, d_out, N, r);
    else if (bx == 32 && by ==  8) launch_shape<32,  8>(d_in, d_out, N, r);
    else if (bx == 32 && by == 16) launch_shape<32, 16>(d_in, d_out, N, r);
    else if (bx == 32 && by == 32) launch_shape<32, 32>(d_in, d_out, N, r);
    else if (bx ==  8 && by ==  8) launch_shape< 8,  8>(d_in, d_out, N, r);
    else {
        fprintf(stderr, "Unsupported block shape %dx%d. Use 8x8, 16x16, 32x8, 32x16 or 32x32.\n", bx, by);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    return run_experiment("cuda_tiled_shared", launch_tiled, argc, argv);
}
