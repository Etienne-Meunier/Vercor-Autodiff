"""Report 23 figure 1: T/S-difference loss landscape with the four descents on it.

Report-22's `paper_figures.py` figure 1, drawn for the tsdiff_avg loss. Reads only
.npy -- no model, no cluster:

    figures/report-23-tsdiff-paper/{loss_grid,ck_grid,ceps_grid}.npy
    figures/report-23-tsdiff-paper200/start_{0..3}/trajectory.npy

Left   the 16x16 forward-only scan, as loss / lowest sampled node on a log scale,
       with each descent's path, start (circle) and final iterate (square).
Right  each run's loss against iteration, raw and rolling-median smoothed.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
GRIDDIR = os.path.join(FIG, "report-23-tsdiff-paper")
PAPER = os.path.join(FIG, "report-23-tsdiff-paper200")

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
STARTS = [(0.05, 0.40), (0.15, 1.00), (0.05, 1.00), (0.15, 0.40)]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
SMOOTH_WINDOW = int(os.environ.get("REPORT23_LOSS_SMOOTH", 9))

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})


def smooth_loss(y, window=SMOOTH_WINDOW):
    """Rolling median of the loss in log space, then a short moving average.

    Report-22's smoother, unchanged: median so that a single deep iterate cannot
    drag the curve down, log space because the loss spans decades, shrinking
    windows at the edges so the curve starts and ends on real data.
    """
    y = np.asarray(y, float)
    out = np.empty_like(y)
    half = window // 2
    logy = np.log10(np.maximum(y, np.finfo(float).tiny))
    for i in range(y.size):
        lo, hi = max(0, i - half), min(y.size, i + half + 1)
        out[i] = np.median(logy[lo:hi])
    smoothed = np.empty_like(out)
    for i in range(out.size):
        lo, hi = max(0, i - 2), min(out.size, i + 3)
        smoothed[i] = out[lo:hi].mean()
    return 10.0 ** smoothed


grid = np.load(os.path.join(GRIDDIR, "loss_grid.npy"))
ck = np.load(os.path.join(GRIDDIR, "ck_grid.npy"))
ceps = np.load(os.path.join(GRIDDIR, "ceps_grid.npy"))
trajs = [np.load(os.path.join(PAPER, f"start_{i}", "trajectory.npy")) for i in range(len(STARTS))]

fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

rel = grid / np.nanmin(grid)
im = axs[0].pcolormesh(ck, ceps, rel.T, cmap="viridis_r", shading="nearest",
                       norm=LogNorm(vmin=1.0, vmax=np.nanmax(rel)))
fig.colorbar(im, ax=axs[0], label="loss / lowest sampled loss")

for traj, color in zip(trajs, COLORS):
    axs[0].plot(traj[:, 0], traj[:, 1], "--", color=color, lw=1.2, alpha=0.45, zorder=3)
    axs[0].plot(traj[0, 0], traj[0, 1], "o", color=color, ms=8, markeredgecolor="white",
                markeredgewidth=1.2, zorder=6)
    axs[0].plot(traj[-1, 0], traj[-1, 1], "s", color=color, ms=9, markeredgecolor="white",
                markeredgewidth=1.2, zorder=6)

axs[0].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=26, markeredgecolor="black",
            markeredgewidth=1.0, zorder=4)
axs[0].annotate("true", xy=(TRUE_C_K, TRUE_C_EPS), xytext=(10, -16),
                textcoords="offset points", fontsize=10, fontweight="bold")
axs[0].set_xlabel("$c_k$")
axs[0].set_ylabel(r"$c_\epsilon$")
axs[0].set_title("Loss landscape and gradient descent paths\n"
                 "(circles: starts, squares: final iterates)")

for i, (traj, color) in enumerate(zip(trajs, COLORS)):
    losses = traj[1:, 2]
    it = np.arange(losses.size)
    axs[1].plot(it, losses, "-", color=color, lw=0.8, alpha=0.25)
    axs[1].plot(it, smooth_loss(losses), "-", color=color, lw=2.0,
                label=f"start {i}: ({STARTS[i][0]}, {STARTS[i][1]:.2f})")
axs[1].set_yscale("log")
# Full range here, unlike report-22: start 3's descent to ~1e-6 is its converged
# state rather than one lucky iterate, so clipping the axis would hide the result.
finite = np.concatenate([t[1:, 2][np.isfinite(t[1:, 2])] for t in trajs])
axs[1].set_ylim(float(finite.min()) / 2, float(finite.max()) * 1.6)
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("loss  (variance-normalised)")
axs[1].set_title(f"Loss during optimization\n(bold: rolling median over {SMOOTH_WINDOW} iterations)")
axs[1].legend(fontsize=8.5)
axs[1].grid(alpha=0.3, which="both")

fig.tight_layout()
p1 = os.path.join(PAPER, "figure1_landscape_multistart.png")
fig.savefig(p1, dpi=200, bbox_inches="tight")
fig.savefig(p1.replace(".png", ".pdf"), bbox_inches="tight")
print(f"saved {p1} (+ .pdf)")

print(f"\n{'start':>5s} {'from':>16s} {'final c_k':>10s} {'final c_eps':>12s} "
      f"{'best c_k':>9s} {'best c_eps':>11s} {'final loss':>12s}")
for i, traj in enumerate(trajs):
    k = int(np.nanargmin(traj[1:, 2]))
    print(f"{i:5d} {'(%.2f, %.2f)' % STARTS[i]:>16s} {traj[-1, 0]:10.4f} {traj[-1, 1]:12.4f} "
          f"{traj[k, 0]:9.4f} {traj[k, 1]:11.4f} {traj[-1, 2]:12.4e}")
