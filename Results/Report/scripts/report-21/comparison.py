"""Report 21's comparison figure: the four observables side by side.

Reads checkpoints only -- no model, so this runs locally in seconds:

    figures/report-21-{mld,temp_avg,mld_avg}/trajectory.npy
        (c_k, c_eps, loss, dL/dc_k, dL/dc_eps) per row, row k+1 holding the
        parameters produced after iteration k and the loss evaluated before it.
    figures/report-20-clip/iterations.txt
        report-20's run in the same layout. Its trajectory.npy predates gradient
        checkpointing and stores only (c_k, c_eps, loss), so the gradients come
        from the iteration table copied out of its job log.

Gradient norms are plotted as a ratio to each run's own median. The four losses
are in different units (K^2 for temperature, m^2 for MLD) and differ by four
orders of magnitude in absolute gradient, so only the shape of the distribution
is comparable across runs -- which is exactly the question: does averaging the
observable over the last three days flatten the outlier tail that report-20 had
to clip?
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
OUT_DIR = os.path.join(FIGURES, "report-21")
os.makedirs(OUT_DIR, exist_ok=True)

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4


def load_report21(name):
    return np.load(os.path.join(FIGURES, f"report-21-{name}", "trajectory.npy"))


def load_report20():
    """report-20's run, reassembled into report-21's 5-column layout."""
    tab = np.loadtxt(os.path.join(FIGURES, "report-20-clip", "iterations.txt"))
    rows = np.column_stack([tab[:, 2], tab[:, 3], tab[:, 1], tab[:, 4], tab[:, 5]])
    start = np.array([[INIT_C_K, INIT_C_EPS, np.nan, np.nan, np.nan]])
    return np.vstack([start, rows])


RUNS = [
    ("surface temp, day 10 (report-20)", load_report20(), "tab:red", "--"),
    ("surface temp, mean days 8-10", load_report21("temp_avg"), "tab:red", "-"),
    ("MLD, day 10", load_report21("mld"), "tab:blue", "--"),
    ("MLD, mean days 8-10", load_report21("mld_avg"), "tab:blue", "-"),
]

fig, axs = plt.subplots(2, 2, figsize=(13, 9))

for label, traj, color, style in RUNS:
    it = np.arange(traj.shape[0])
    axs[0, 0].plot(it, traj[:, 1], style, color=color, lw=1.8, label=label)
    axs[0, 1].plot(it, traj[:, 0], style, color=color, lw=1.8, label=label)

    g = traj[1:, 3:5]
    gn = np.linalg.norm(g, axis=1)
    axs[1, 0].plot(np.arange(gn.size), gn / np.median(gn), style, color=color, lw=1.2, alpha=0.85, label=label)

    axs[1, 1].plot(traj[:, 0], traj[:, 1], style, color=color, lw=1.5, alpha=0.85, label=label)
    axs[1, 1].plot(traj[-1, 0], traj[-1, 1], "s", color=color, ms=8)

axs[0, 0].axhline(TRUE_C_EPS, color="gold", lw=2.5, zorder=0, label="true c_eps = 0.7")
axs[0, 0].set_xlabel("iteration")
axs[0, 0].set_ylabel("c_eps")
axs[0, 0].set_title("c_eps: only the time-averaged runs reach truth")
axs[0, 0].legend(fontsize=8, loc="lower right")
axs[0, 0].grid(alpha=0.3)

axs[0, 1].axhline(TRUE_C_K, color="gold", lw=2.5, zorder=0, label="true c_k = 0.1")
axs[0, 1].set_xlabel("iteration")
axs[0, 1].set_ylabel("c_k")
axs[0, 1].set_title("c_k: every run gets close")
axs[0, 1].legend(fontsize=8, loc="lower right")
axs[0, 1].grid(alpha=0.3)

axs[1, 0].axhline(1.0, color="black", lw=1, ls=":")
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("iteration")
axs[1, 0].set_ylabel("|g| / median |g| of the same run")
axs[1, 0].set_title("Gradient norm relative to each run's own median")
axs[1, 0].legend(fontsize=8)
axs[1, 0].grid(alpha=0.3)

axs[1, 1].plot(INIT_C_K, INIT_C_EPS, "o", color="black", ms=9, label="start")
axs[1, 1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=22, markeredgecolor="black", zorder=5, label="true")
axs[1, 1].set_xlabel("c_k")
axs[1, 1].set_ylabel("c_eps")
axs[1, 1].set_title("Parameter trajectories (squares = final iterate)")
axs[1, 1].legend(fontsize=8, loc="upper left")
axs[1, 1].grid(alpha=0.3)

fig.suptitle("Report 21: instantaneous vs time-averaged observables, 10-day coupled rollout")
fig.tight_layout()
path = os.path.join(OUT_DIR, "comparison.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"saved {path}")

print(f"\n{'run':34s} {'final c_k':>10s} {'final c_eps':>12s} {'best c_k':>9s} {'best c_eps':>11s} "
      f"{'max/med |g|':>12s} {'>10x med':>9s} {'reversals':>10s}")
for label, traj, _, _ in RUNS:
    losses = traj[1:, 2]
    k = int(np.nanargmin(losses))          # row k+1 holds the loss evaluated at row k's params
    g = traj[1:, 3:5]
    gn = np.linalg.norm(g, axis=1)
    u = g / np.where(gn[:, None] == 0, 1.0, gn[:, None])
    cos = np.sum(u[1:] * u[:-1], axis=1)
    print(f"{label:34s} {traj[-1, 0]:10.4f} {traj[-1, 1]:12.4f} {traj[k, 0]:9.4f} {traj[k, 1]:11.4f} "
          f"{gn.max() / np.median(gn):11.1f}x {int((gn > 10 * np.median(gn)).sum()):9d} "
          f"{int((cos < 0).sum()):6d}/{cos.size:<4d}")
