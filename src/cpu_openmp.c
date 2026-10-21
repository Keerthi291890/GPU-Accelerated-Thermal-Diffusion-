
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "../include/simulation_config.h"

#define INDEX(row, col) ((row) * GRID_SIZE + (col))


static void initialize_grid(float *grid)
{
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        grid[i] = AMBIENT_TEMP;
    }

    int center = GRID_SIZE / 2;

    for (int i = center - HOTSPOT_RADIUS;
         i < center + HOTSPOT_RADIUS; i++) {
        for (int j = center - HOTSPOT_RADIUS;
             j < center + HOTSPOT_RADIUS; j++) {
            grid[i * GRID_SIZE + j] = HOT_TEMP;
        }
    }
}

static void write_grid_csv(const char *filename, const float *grid)
{
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    for (int row = 0; row < GRID_SIZE; row++) {
        for (int col = 0; col < GRID_SIZE; col++) {
            if (col > 0) {
                fprintf(file, ",");
            }

            fprintf(file, "%.9g", grid[INDEX(row, col)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(void)
{
    const size_t total_cells =
        (size_t)GRID_SIZE * GRID_SIZE;
    const size_t bytes = total_cells * sizeof(float);

    float *current = malloc(bytes);
    float *next = malloc(bytes);

    if (current == NULL || next == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    initialize_grid(current);
    for (size_t i = 0; i < total_cells; i++) {
    next[i] = AMBIENT_TEMP;
}

    /* Output directory already exists. */

    write_grid_csv("results/cpu/initial_temperature.csv", current);

    double start = omp_get_wtime();

    
#pragma omp parallel shared(current, next)
{
    for (int step = 0; step < TIME_STEPS; step++) {

        #pragma omp for schedule(static)
        for (int row = 1; row < GRID_SIZE - 1; row++) {
            for (int col = 1; col < GRID_SIZE - 1; col++) {
                int idx = INDEX(row, col);

                float center = current[idx];
                float up     = current[INDEX(row - 1, col)];
                float down   = current[INDEX(row + 1, col)];
                float left   = current[INDEX(row, col - 1)];
                float right  = current[INDEX(row, col + 1)];

                next[idx] = center + DIFFUSION_R *
                    (up + down + left + right - 4.0f * center);
            }
        }

        #pragma omp single
        {
            float *temp = current;
            current = next;
            next = temp;
        }
    }
}


    double end = omp_get_wtime();
    double runtime = end - start;

    write_grid_csv("results/cpu/final_openmp.csv", current);

    FILE *runtime_file =
        fopen("results/cpu/runtime_openmp.csv", "w");

    if (runtime_file == NULL) {
        perror("results/cpu/runtime_openmp.csv");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    fprintf(runtime_file,
            "implementation,grid_size,time_steps,runtime_seconds\n");
    fprintf(runtime_file, "openmp,%d,%d,%.9f\n",
            GRID_SIZE, TIME_STEPS, runtime);
    fclose(runtime_file);

    printf("Implementation: OpenMP C\n");
    printf("Grid: %d x %d\n", GRID_SIZE, GRID_SIZE);
    printf("Timesteps: %d\n", TIME_STEPS);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("Runtime: %.6f seconds\n", runtime);
    printf("Center temperature: %.6f\n",
           current[INDEX(GRID_SIZE / 2, GRID_SIZE / 2)]);

    free(current);
    free(next);

    return EXIT_SUCCESS;
}
