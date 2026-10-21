
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "cpu"


def load_grid(filename):
    """Load a temperature grid from a CSV file."""
    path = RESULTS_DIR / filename

    with path.open("r", newline="") as file:
        return np.array([
            [float(value) for value in row]
            for row in csv.reader(file)
        ])


# Load initial and final temperature distributions
initial = load_grid("initial_temperature.csv")
final = load_grid("final_serial.csv")

# Plot both distributions using the same temperature scale

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6),
    constrained_layout=True
)

for ax, grid, title in zip(
    axes,
    [initial, final],
    ["Initial Temperature", "Final Temperature (1000 Steps)"]
):
    image = ax.imshow(
        grid,
        origin="lower",
        cmap="inferno",
        vmin=25,
        vmax=100,
        interpolation="nearest"
    )

    ax.set_title(title)
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")

fig.colorbar(
    image,
    ax=axes,
    label="Temperature (°C)",
    shrink=0.85,
    pad=0.04
)

fig.suptitle("Thermal Diffusion: Initial vs Final State")

output_path = RESULTS_DIR / "temperature_visualization.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Visualization saved to: {output_path}")

plt.show()
