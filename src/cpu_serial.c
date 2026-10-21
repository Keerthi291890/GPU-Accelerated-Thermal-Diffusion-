#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/simulation_config.h"

#define TOTAL_CELLS (GRID_SIZE * GRID_SIZE)

static void initialize_grid(float *grid) {
    for (int i = 0; i < TOTAL_CELLS; i++) {
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

static void save_csv(const char *filename,
                     const float *grid) {
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror(filename);
        return;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (j > 0) {
                fprintf(file, ",");
            }

            fprintf(file, "%.9g",
                    grid[i * GRID_SIZE + j]);
        }

        fprintf(file, "\n");
    }

    fclose(file);
}

static double wall_time_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return (double)ts.tv_sec
         + (double)ts.tv_nsec / 1e9;
}

static void simulate(float *current, float *next) {
    for (int step = 0; step < TIME_STEPS; step++) {

        /*
         * Reset next grid so that its boundary
         * cells remain at the ambient temperature.
         */
        for (int i = 0; i < TOTAL_CELLS; i++) {
            next[i] = AMBIENT_TEMP;
        }

        /* Update interior cells only. */
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {

                int idx = i * GRID_SIZE + j;

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

        /* Exchange the old and new grids. */
        float *temp = current;
        current = next;
        next = temp;
    }

    /*
     * The local pointer swap above does not change
     * the caller's pointer. Copy the final result
     * back into the original current buffer.
     */
    if (TIME_STEPS % 2 != 0) {
        for (int i = 0; i < TOTAL_CELLS; i++) {
            current[i] = next[i];
        }
    }
}

int main(void) {
    float *current =
        malloc(TOTAL_CELLS * sizeof(float));
    float *next =
        malloc(TOTAL_CELLS * sizeof(float));

    if (current == NULL || next == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    initialize_grid(current);

    if (GRID_SIZE < 3 ||
        DIFFUSION_R <= 0.0f ||
        DIFFUSION_R > 0.25f) {
        fprintf(stderr, "Invalid simulation parameters.\n");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    if (TIME_STEPS % 2 != 0) {
        fprintf(stderr,
                "This implementation expects an even "
                "number of timesteps.\n");
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    if (system("mkdir results\\cpu 2>nul") != 0) {
        /* Directory creation is best-effort on Windows. */
    }

    save_csv("results/cpu/initial_temperature.csv",
             current);

    double start = wall_time_seconds();

    simulate(current, next);

    double end = wall_time_seconds();
    double elapsed = end - start;

    save_csv("results/cpu/final_serial.csv", current);

    int center_idx =
        (GRID_SIZE / 2) * GRID_SIZE + GRID_SIZE / 2;

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
    }

    free(current);
    free(next);

    return EXIT_SUCCESS;
}
