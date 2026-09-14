"""Compare the four landscapes of the 2x2 campaign: observable x SST restoring.

    python scripts/report-24/compare_grids.py

Each panel is normalised by its own lowest sampled node and drawn in log10. Local
minima are nodes strictly below all their neighbours.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "figures", "report-24")
CONFIGS = [("mld_avg_restore", "MLD, restoring on"),
           ("mld_avg_norestore", "MLD, restoring off"),
           ("tsdiff_avg_restore", "T/S differences, restoring on"),
           ("tsdiff_avg_norestore", "T/S differences, restoring off")]
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7


def local_minima(grid):
    n0, n1 = grid.shape
    found = []
    for i in range(n0):
        for j in range(n1):
            neighbours = [grid[a, b]
                          for a in range(max(i - 1, 0), min(i + 2, n0))
                          for b in range(max(j - 1, 0), min(j + 2, n1)) if (a, b) != (i, j)]
            if all(grid[i, j] < x for x in neighbours):
                found.append((i, j, grid[i, j]))
    return sorted(found, key=lambda t: t[2])


figure, axes = plt.subplots(1, 4, figsize=(21, 4.8), sharey=True)
for axis, (name, title) in zip(axes, CONFIGS):
    directory = os.path.join(FIG, name, "grid")
    grid = np.load(os.path.join(directory, "loss_grid.npy"))
    c_k = np.load(os.path.join(directory, "c_k_values.npy"))
    c_eps = np.load(os.path.join(directory, "c_eps_values.npy"))
    minima = local_minima(grid)

    print(f"\n{title}: range {np.nanmin(grid):.4e} .. {np.nanmax(grid):.4e} "
          f"({np.nanmax(grid) / np.nanmin(grid):.0f}x), {len(minima)} local minima")
    for rank, (i, j, value) in enumerate(minima[:6]):
        print(f"  {rank}: c_k={c_k[i]:.4f} c_eps={c_eps[j]:.4f} loss={value:.4e} "
              f"({value / minima[0][2]:.2f}x)")

    relative = np.log10(grid / np.nanmin(grid))
    mesh = axis.pcolormesh(c_k, c_eps, relative.T, shading="nearest", cmap="viridis")
    figure.colorbar(mesh, ax=axis, label="log10(loss / lowest node)")
    for rank, (i, j, _) in enumerate(minima):
        axis.plot(c_k[i], c_eps[j], "o" if rank == 0 else "s", mfc="none", mec="red", ms=9, mew=1.5)
    axis.plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=18, mec="black")
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("$c_k$")
axes[0].set_ylabel(r"$c_\epsilon$")
figure.suptitle("Report 24: loss landscapes, observable x SST restoring "
                "(circle: lowest node, squares: other local minima, star: truth)")
figure.tight_layout()
path = os.path.join(FIG, "landscapes.png")
figure.savefig(path, dpi=150)
print(f"\nsaved {path}")
