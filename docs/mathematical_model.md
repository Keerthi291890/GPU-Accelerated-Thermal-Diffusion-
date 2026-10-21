# Mathematical Model and CPU Baselines

## 1. Overview

This project studies two-dimensional thermal diffusion using a numerical simulation and evaluates parallel computing techniques for accelerating the computation.

The objective of this component is to establish a reliable CPU reference implementation using serial C and OpenMP C. These implementations solve the same mathematical problem and provide reference outputs for validating and comparing subsequent GPU implementations.

The simulation models heat spreading from a localized central hotspot into a cooler surrounding region over a fixed number of timesteps.

## 2. Mathematical Model

### 2.1 Heat Equation

Thermal diffusion in a two-dimensional domain is described by the heat equation:

\[
\frac{\partial T}{\partial t}
=
\alpha\left(
\frac{\partial^2 T}{\partial x^2}
+
\frac{\partial^2 T}{\partial y^2}
\right)
\]

where:

- \(T(x,y,t)\) is the temperature at position \((x,y)\) and time \(t\).
- \(t\) represents simulation time.
- \(x\) and \(y\) represent spatial coordinates.
- \(\alpha\) is the thermal diffusivity of the material.

The equation describes how temperature evolves over time due to heat transfer between neighboring regions.

### 2.2 Finite-Difference Discretization

To solve the equation computationally, the continuous domain is represented as a two-dimensional grid.

The spatial derivatives are approximated using the temperatures of neighboring cells. Assuming equal spacing in both spatial directions, the explicit finite-difference update equation is:

\[
\begin{aligned}
T_{i,j}^{n+1}=T_{i,j}^{n}+r(&T_{i-1,j}^{n}
+T_{i+1,j}^{n}\\
&+T_{i,j-1}^{n}
+T_{i,j+1}^{n}\\
&-4T_{i,j}^{n})
\end{aligned}
\]

where:

- \(T_{i,j}^{n}\) is the temperature at row \(i\), column \(j\), at timestep \(n\).
- \(T_{i,j}^{n+1}\) is the temperature at the next timestep.
- The four neighboring cells are the cells immediately above, below, left, and right of the current cell.
- \(r\) is the dimensionless diffusion parameter.

The diffusion parameter is defined as:

\[
r=\frac{\alpha\Delta t}{\Delta x^2}
\]

Here, \(\Delta t\) is the timestep size and \(\Delta x\) is the uniform grid spacing.

In the implementation, the update is calculated as:

```c
next[idx] = center + DIFFUSION_R *
    (up + down + left + right - 4.0f * center);
```

This operation is the central computational kernel of the simulation.

## 3. Simulation Configuration

The simulation uses the following configuration:

| Parameter | Value |
|---|---:|
| Grid dimensions | 1024 × 1024 |
| Total grid cells | 1,048,576 |
| Number of timesteps | 1,000 |
| Ambient temperature | 25°C |
| Initial hotspot temperature | 100°C |
| Hotspot radius parameter | 25 cells |
| Diffusion parameter, \(r\) | 0.2 |
| Boundary condition | Fixed temperature |
| Programming language | C |
| CPU implementations | Serial C and OpenMP C |

The initial grid is filled with the ambient temperature. A square hotspot is then initialized at the center of the grid.

The hotspot is created using a row and column range extending from the center minus the hotspot radius to the center plus the hotspot radius, with the upper limit excluded. Consequently, the initialized hotspot occupies 50 × 50 cells.

The same configuration and initialization logic are used for both CPU implementations to maintain consistency.

## 4. Numerical Stability

For the two-dimensional explicit finite-difference scheme with equal spatial spacing, the standard stability condition is:

\[
r\leq\frac{1}{4}
\]

The selected value is:

\[
r=0.2
\]

Since \(0.2\leq0.25\), the selected parameter satisfies the standard stability condition.

The update equation can also be expressed as:

\[
\begin{aligned}
T_{i,j}^{n+1}
={}&(1-4r)T_{i,j}^{n}\\
&+r(T_{\mathrm{up}}+T_{\mathrm{down}}
+T_{\mathrm{left}}+T_{\mathrm{right}})
\end{aligned}
\]

For \(r=0.2\), the coefficient of the current cell and the coefficient of each neighboring cell are all 0.2. These coefficients are nonnegative and sum to one.

Therefore, each updated interior temperature is a weighted average of the current cell and its four neighbors. This supports the expected smoothing behavior of the diffusion process.

## 5. Boundary Conditions

The simulation uses fixed-temperature, or Dirichlet, boundary conditions.

All cells along the outermost rows and columns are maintained at the ambient temperature of 25°C.

Only interior cells are updated during each timestep. Before the interior update, the next grid is initialized to the ambient temperature. The interior temperatures are then calculated from the current grid.

This models a plate whose edges are maintained at a constant temperature. It does not represent an insulated boundary, which would instead require a zero-heat-flux condition.

## 6. CPU Implementations

### 6.1 Serial C Implementation

The serial implementation performs the simulation sequentially.

For each timestep, it:

1. Initializes the next grid to the ambient temperature.
2. Iterates through the interior cells.
3. Calculates each cell's temperature using the finite-difference equation.
4. Stores the result in the next grid.
5. Swaps the current and next grid pointers.

The simulation repeats these operations for 1,000 timesteps.

The serial implementation serves as the CPU reference for numerical correctness and performance comparisons.

### 6.2 OpenMP C Implementation

The OpenMP implementation parallelizes the interior grid computation using multiple CPU threads.

The interior rows are distributed among threads using:

```c
#pragma omp for schedule(static)
```

Each thread computes the temperatures for its assigned cells. All threads read from the current grid and write to distinct locations in the next grid.

A synchronization barrier ensures that all cell updates finish before the grid pointers are swapped. The pointer swap is performed within an OpenMP `single` region, followed by synchronization before the next timestep.

This double-buffering strategy avoids updating values in place and ensures that every cell at a given timestep uses temperatures from the same previous timestep.

Both implementations use the same numerical update equation and boundary conditions.

## 7. Correctness Validation

The final temperature grids produced by the serial and OpenMP implementations were compared cell by cell.

The validation used a grid size of 1024 × 1024 and 1,000 timesteps.

| Validation metric | Result |
|---|---:|
| Grid dimensions | 1024 × 1024 |
| Cells compared | 1,048,576 |
| Different cells | 0 |
| Maximum absolute difference | 0 |

The comparison found no differing cells and a maximum absolute difference of zero.

This confirms that the two CPU implementations produced identical final grids for the tested configuration.

The center temperature was also reported as approximately 71.636833°C by both implementations.

## 8. Temperature Visualization

A Python visualization script using NumPy and Matplotlib was used to generate a heatmap comparing the initial and final temperature distributions.

The visualization illustrates:

- The initial square hotspot at the center of the grid.
- Heat spreading from the hotspot into the surrounding region.
- The reduction in the peak temperature over time.
- The fixed ambient-temperature boundary conditions.

The resulting figure is saved as:

`results/cpu/temperature_visualization.png`

The visualization provides a qualitative check that the simulation exhibits the expected diffusion behavior.

## 9. CPU Performance Baselines

The simulation was benchmarked using the same grid dimensions and timestep count for each implementation.

The following results are based on five-run average timings collected during development.

| Implementation | Threads | Average runtime (s) | Speedup |
|---|---:|---:|---:|
| Serial C | 1 | 0.603436 | 1.00× |
| OpenMP C | 2 | 0.408000 | 1.48× |
| OpenMP C | 4 | 0.426400 | 1.42× |

Speedup is calculated as:

\[
S=\frac{T_{\mathrm{serial}}}{T_{\mathrm{parallel}}}
\]

For the two-thread OpenMP configuration:

\[
S=\frac{0.603436}{0.408000}\approx1.48
\]

The two-thread configuration achieved approximately 1.48× speedup relative to the serial baseline. The four-thread configuration achieved approximately 1.42× speedup.

These measurements show that OpenMP improved runtime for the tested workload. However, the four-thread result was slower than the two-thread result in these measurements. Thread scheduling, synchronization, memory-bandwidth limitations, CPU load, and measurement variability may contribute to this behavior.

The results are initial CPU baselines rather than a definitive characterization of the processor's maximum parallel performance. Further controlled repetitions would be required for a rigorous performance study.

## 10. Conclusion

The mathematical model and CPU baseline implementations establish the foundation for the thermal diffusion case study.

The serial C implementation provides a reference solution, while the OpenMP implementation demonstrates parallel execution on the CPU. Exact agreement between their final temperature grids validates the OpenMP implementation for the tested configuration.

The visualization confirms the expected qualitative behavior of heat diffusion, and the benchmark measurements establish initial performance baselines for subsequent CPU–GPU comparisons.

The same mathematical model, grid dimensions, initial conditions, boundary conditions, and timestep count should be retained when validating and comparing GPU implementations.
