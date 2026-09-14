"""Report 18: the loss grids drawn as raw cells -- no interpolation.

`contourf` smooths between the 12 sampled values per axis, which moves the
apparent dark region away from the sampled nodes and makes it hard to tell
whether the true parameters really sit at the minimum. This draws one block per
grid node (shading="nearest"), prints every value, and marks both the true
parameters and the actual argmin cell.

Axis convention: report-8 stores loss_grid[j, i] with j over c_eps (so it is
transposed on load); report-18 stores loss_grid[i, j] with i over c_k.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
OUT_DIR = os.path.join(FIG, "report-18")

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
PANELS = [("report-8", True, "Report 8: full-column temperature loss"),
          ("report-18", False, "Report 18: mean(|dT/dz|) top-3 loss")]

fig, axs = plt.subplots(1, 2, figsize=(21, 9))
for ax, (rep, transpose, title) in zip(axs, PANELS):
    d = os.path.join(FIG, rep)
    g = np.load(os.path.join(d, "loss_grid.npy"))
    ck = np.load(os.path.join(d, "ck_grid.npy"))
    ce = np.load(os.path.join(d, "ceps_grid.npy"))
    if transpose:
        g = g.T

    rel = np.log10(g / np.nanmin(g))
    # shading="nearest" puts one flat block per sampled node, centred on it:
    # every colour on screen is a number that was actually computed.
    im = ax.pcolormesh(ck, ce, rel.T, cmap="viridis", shading="nearest")
    fig.colorbar(im, ax=ax, label="log10(loss / min loss)")

    for i, x in enumerate(ck):
        for j, y in enumerate(ce):
            ax.text(x, y, f"{rel[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if rel[i, j] < 1.0 else "black")

    gi, gj = np.unravel_index(np.nanargmin(g), g.shape)
    ax.plot(ck[gi], ce[gj], "s", ms=26, mfc="none", mec="red", mew=2.5, zorder=5,
            label=f"grid min: ({ck[gi]:.4f}, {ce[gj]:.4f})")
    i0, j0 = int(np.argmin(abs(ck - TRUE_C_K))), int(np.argmin(abs(ce - TRUE_C_EPS)))
    ax.plot(ck[i0], ce[j0], "s", ms=26, mfc="none", mec="white", mew=2.5, zorder=5,
            label=f"node nearest truth: {g[i0, j0] / np.nanmin(g):.2f}x the min")
    ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="white", ms=26, markeredgecolor="black",
            zorder=6, label=f"true ({TRUE_C_K}, {TRUE_C_EPS})")

    ax.set_xticks(ck)
    ax.set_xticklabels([f"{v:.4f}" for v in ck], rotation=90, fontsize=7)
    ax.set_yticks(ce)
    ax.set_yticklabels([f"{v:.4f}" for v in ce], fontsize=7)
    ax.set_xlabel("c_k (sampled values only)")
    ax.set_ylabel("c_eps (sampled values only)")
    ax.set_title(f"{title}\nmin loss = {np.nanmin(g):.4e}; cells show log10(loss / min)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

fig.suptitle("Report 18: raw 12x12 loss grids, no interpolation -- one block per evaluated point")
fig.tight_layout()
p = os.path.join(OUT_DIR, "raw_grid.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
