
#ifndef SIMULATION_CONFIG_H
#define SIMULATION_CONFIG_H

#ifndef GRID_SIZE
#define GRID_SIZE 1024
#endif

#ifndef TIME_STEPS
#define TIME_STEPS 1000
#endif

#define AMBIENT_TEMP 25.0f
#define HOT_TEMP 100.0f

#define HOTSPOT_RADIUS 25

/*
 * Finite-difference coefficient:
 * r = alpha * dt / (dx * dx)
 *
 * For the explicit 2D scheme, r <= 0.25.
 */
#define DIFFUSION_R 0.2f

#endif
