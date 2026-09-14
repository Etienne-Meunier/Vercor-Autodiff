"""Report 20: loss and fitted-parameter evolution, drawn from trajectory.npy.

No model runs -- reads the checkpointed trajectory, so it redraws in a second.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
OUT_DIR = os.environ.get("REPORT20_OUT_DIR", os.path.join(FIG, "report-20-clip"))

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.40
R15 = (0.0876, 0.4892)  # report-15's N=20 result, same recipe

traj = np.load(os.path.join(OUT_DIR, "trajectory.npy"))
ck, ce, loss = traj[:, 0], traj[:, 1], traj[:, 2]
it = np.arange(len(traj))
runmin = np.fmin.accumulate(np.where(np.isnan(loss), np.inf, loss))

fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))

ax = axs[0]
ax.semilogy(it[1:], loss[1:], "-o", color="darkgreen", ms=3.5, lw=1, alpha=0.55, label="loss")
ax.semilogy(it[1:], runmin[1:], "-", color="black", lw=2, label="running minimum")
ax.set_xlabel("iteration")
ax.set_ylabel("loss: masked surface-temperature SSE")
ax.set_title("Loss")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)

for ax, vals, true, init, r15, name in (
    (axs[1], ck, TRUE_C_K, INIT_C_K, R15[0], "c_k"),
    (axs[2], ce, TRUE_C_EPS, INIT_C_EPS, R15[1], "c_eps"),
):
    ax.plot(it, vals, "-o", color="darkgreen", ms=3.5, lw=1.4, label=f"{name} (N=10, clipped)")
    ax.axhline(true, color="goldenrod", ls="--", lw=2, label=f"true {name} = {true}")
    ax.axhline(r15, color="tab:red", ls=":", lw=1.8, label=f"report-15 final (N=20) = {r15:.4f}")
    ax.plot(0, init, "o", color="black", ms=9, label=f"start = {init}")
    err = 100 * abs(vals[-1] - true) / true
    ax.set_xlabel("iteration")
    ax.set_ylabel(name)
    ax.set_title(f"{name}: {init} -> {vals[-1]:.4f}   ({err:.1f}% error)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Report 20: gradient-based calibration of a coupled ocean-atmosphere model -- "
             "reverse-mode AD through a 10-day OCN+LND+ATM rollout", fontsize=13)
fig.tight_layout()
p = os.path.join(OUT_DIR, "convergence.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
