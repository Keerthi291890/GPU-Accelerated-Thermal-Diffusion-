#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/simulation_config.h"

#define TOTAL_CELLS ((size_t)GRID_SIZE * GRID_SIZE)

static void initialize_grid(float *grid)
{
    for (size_t i = 0; i < TOTAL_CELLS; i++) {
        grid[i] = AMBIENT_TEMP;
    }

    int center = GRID_SIZE / 2;

    for (int i = center - HOTSPOT_RADIUS;
         i < center + HOTSPOT_RADIUS; i++) {
        for (int j = center - HOTSPOT_RADIUS;
             j < center + HOTSPOT_RADIUS; j++) {
            grid[(size_t)i * GRID_SIZE + j] = HOT_TEMP;
        }
    }
}

static int save_csv(const char *filename, const float *grid)
{
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror(filename);
        return 0;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (j > 0) {
                fprintf(file, ",");
            }

            fprintf(file, "%.9g",
                    grid[(size_t)i * GRID_SIZE + j]);
        }

        fprintf(file, "\n");
    }

    fclose(file);
    return 1;
}

static double wall_time_seconds(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return (double)ts.tv_sec
         + (double)ts.tv_nsec / 1e9;
}

static void simulate(float **current_ptr, float **next_ptr)
{
    float *current = *current_ptr;
    float *next = *next_ptr;

    for (int step = 0; step < TIME_STEPS; step++) {

        /* Update every interior cell; preserve boundaries. */
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {

                size_t idx = (size_t)i * GRID_SIZE + j;

                float center = current[idx];
                float up     = current[idx - GRID_SIZE];
                float down   = current[idx + GRID_SIZE];
                float left   = current[idx - 1];
                float right  = current[idx + 1];

                next[idx] = center + DIFFUSION_R *
                    (up + down + left + right
                     - 4.0f * center);
            }
        }

        /* Swap the buffers without copying the grid. */
        float *temp = current;
        current = next;
        next = temp;
    }

    /* Make the final buffer pointers visible to the caller. */
    *current_ptr = current;
    *next_ptr = next;
}

int main(void)
{
    if (GRID_SIZE < 3 ||
        GRID_SIZE < 2 * HOTSPOT_RADIUS ||
        TIME_STEPS < 0 ||
        DIFFUSION_R <= 0.0f ||
        DIFFUSION_R > 0.25f) {
        fprintf(stderr, "Invalid simulation parameters.\n");
        return EXIT_FAILURE;
    }

    float *current = malloc(TOTAL_CELLS * sizeof(float));
    float *next = malloc(TOTAL_CELLS * sizeof(float));

    if (current == NULL || next == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    initialize_grid(current);
    initialize_grid(next);

    /* Both buffers start with ambient-temperature boundaries.
       The hotspot is applied only to the initial current grid. */
    if (system("mkdir results\\cpu 2>nul") != 0) {
        /* Directory creation is best-effort on Windows. */
    }

    if (!save_csv("results/cpu/initial_temperature.csv", current)) {
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    double start = wall_time_seconds();

    simulate(&current, &next);

    double end = wall_time_seconds();
    double elapsed = end - start;

    if (!save_csv("results/cpu/final_serial.csv", current)) {
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    size_t center_idx =
        (size_t)(GRID_SIZE / 2) * GRID_SIZE + GRID_SIZE / 2;

    printf("Implementation: Serial C\n");
    printf("Grid: %d x %d\n", GRID_SIZE, GRID_SIZE);
    printf("Timesteps: %d\n", TIME_STEPS);
    printf("Runtime: %.6f seconds\n", elapsed);
    printf("Center temperature: %.6f\n",
           current[center_idx]);

    FILE *runtime_file =
        fopen("results/cpu/runtime_serial.csv", "w");

    if (runtime_file != NULL) {
        fprintf(runtime_file,
                "implementation,grid_size,time_steps,"
                "runtime_seconds\n");
        fprintf(runtime_file, "serial,%d,%d,%.9f\n",
                GRID_SIZE, TIME_STEPS, elapsed);
        fclose(runtime_file);
    } else {
        perror("results/cpu/runtime_serial.csv");
    }

    free(current);
    free(next);

    return EXIT_SUCCESS;
}