"""Report 23: the four 200-iteration descents, T/S differences against report-22's MLD.

Reads only trajectory.npy (rows c_k, c_eps, loss, dL/dc_k, dL/dc_eps; row 0 is the
start), runs locally:

    python descents_compare.py

Panels: c_eps and c_k against iteration (solid = T/S differences, dashed = MLD),
and each run's loss divided by its own start-point loss, so the two observables'
differently-scaled losses share one axis.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
SETS = [("T/S differences", os.path.join(FIG, "report-23-tsdiff-paper200"), "-"),
        ("MLD (report-22)", os.path.join(FIG, "report-21-paper200"), "--")]
STARTS = ["0 (0.05, 0.40)", "1 (0.15, 1.00)", "2 (0.05, 1.00)", "3 (0.15, 0.40)"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
for label, d, ls in SETS:
    for s in range(4):
        t = np.load(os.path.join(d, f"start_{s}", "trajectory.npy"))
        it = np.arange(t.shape[0])
        kw = dict(color=COLORS[s], ls=ls, lw=1.2)
        axs[0].plot(it, t[:, 1], **kw)
        axs[1].plot(it, t[:, 0], **kw)
        axs[2].plot(it[1:] - 1, t[1:, 2] / t[1, 2], **kw)

axs[0].axhline(0.7, color="black", lw=0.8)
axs[1].axhline(0.1, color="black", lw=0.8)
axs[0].set_ylabel("c_eps")
axs[1].set_ylabel("c_k")
axs[2].set_ylabel("loss / start-point loss")
axs[2].set_yscale("log")
for ax, title in zip(axs, ["c_eps (true 0.7)", "c_k (true 0.1)", "loss, normalised by its start value"]):
    ax.set_xlabel("iteration")
    ax.set_title(title)
    ax.grid(alpha=0.3)

handles = [plt.Line2D([], [], color=c, lw=2) for c in COLORS]
handles += [plt.Line2D([], [], color="gray", ls=ls) for _, _, ls in SETS]
axs[0].legend(handles, [f"start {s}" for s in STARTS] + [lab for lab, _, _ in SETS], fontsize=8)
fig.suptitle("Report 23: multistart descents, top-3 T/S differences (solid) vs MLD (dashed), 200 iterations")
fig.tight_layout()
out = os.path.join(FIG, "report-23-tsdiff-paper200", "descents_compare.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
