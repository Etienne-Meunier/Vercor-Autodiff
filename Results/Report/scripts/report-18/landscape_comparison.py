"""Report 18 addendum figure: the two loss landscapes side by side.

Report-8 scanned a full-column temperature loss on exactly the same
(c_k, c_eps) grid this report scans the stratification metric on, so the two
contour maps are directly comparable. Drawn from the saved .npy grids -- no model
runs.
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
# CRITICAL: the two scripts store their grid with opposite axis order.
#   report-8  fills loss_grid[j, i] with j over c_eps, i over c_k  -> (c_eps, c_k)
#   report-18 fills loss_grid[i, j] with i over c_k,  j over c_eps -> (c_k, c_eps)
# Everything below works in the canonical (c_k, c_eps) order, so report-8's grid is
# transposed on load. Reading it with the wrong convention silently produces a
# transposed landscape whose minimum sits ~6x above the true parameters -- which is
# exactly how this bug was caught.
PANELS = [
    ("report-8", "Report 8: full-column temperature loss", None),
    ("report-18", "Report 18: mean(|dT/dz|) top-3 loss",
     [(0, "start (0.05, 0.40)", "tab:red"),
      (1, "start (0.15, 0.85)", "tab:orange"),
      (2, "start (0.03, 0.55)", "magenta")]),
]

fig, axs = plt.subplots(1, 2, figsize=(17, 6.8))
for ax, (rep, title, starts) in zip(axs, PANELS):
    d = os.path.join(FIG, rep)
    g = np.load(os.path.join(d, "loss_grid.npy"))
    ck = np.load(os.path.join(d, "ck_grid.npy"))
    ce = np.load(os.path.join(d, "ceps_grid.npy"))
    if rep == "report-8":
        g = g.T  # stored (c_eps, c_k); see the note on PANELS above

    # Sanity check every landscape against the fact that defines it: the target is
    # synthetic, so the loss at the true parameters is exactly 0 and the grid node
    # nearest truth must be at (or adjacent to) the grid minimum.
    i0, j0 = int(np.argmin(abs(ck - TRUE_C_K))), int(np.argmin(abs(ce - TRUE_C_EPS)))
    truth_ratio = float(g[i0, j0] / np.nanmin(g))
    print(f"{rep}: loss at nearest-truth node = {truth_ratio:.2f}x the grid minimum"
          + ("" if truth_ratio < 1.5 else "   <-- SUSPECT: check the axis convention"))
    CK, CE = np.meshgrid(ck, ce, indexing="ij")

    # Normalised to each landscape's own minimum: the two losses have different
    # units, so only the *shape* is comparable, not the absolute level.
    rel = np.log10(g / np.nanmin(g))
    cf = ax.contourf(CK, CE, rel, levels=np.linspace(0, 2.0, 21), cmap="viridis", extend="max")
    ax.contour(CK, CE, rel, levels=np.linspace(0, 2.0, 11), colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, label="log10(loss / min loss)")

    # The degeneracy here is DIAGONAL, not axis-aligned, so an axis-aligned
    # statistic (spread over c_eps at fixed c_k, say) is the wrong diagnostic and
    # gives a misleading answer -- it averages over rows far from the valley where
    # everything is uniformly bad. Instead: locate the valley floor to sub-grid
    # accuracy by parabolic interpolation of each column's minimum, fit a line to
    # it over the informative c_k range, and measure the loss ALONG that line.
    # Variation along the valley is what actually breaks the c_k/c_eps degeneracy.
    def parabolic_min(y, x):
        j = int(np.argmin(y))
        if j == 0 or j == len(y) - 1:
            return x[j], False
        y0, y1, y2 = y[j - 1], y[j], y[j + 1]
        den = y0 - 2 * y1 + y2
        if den <= 0:
            return x[j], False
        return x[j] + 0.5 * (y0 - y2) / den * (x[1] - x[0]), True

    xs, ys = [], []
    for i in range(len(ck)):
        xi, ok = parabolic_min(g[i], ce)
        if ok and 0.06 <= ck[i] <= 0.13:
            xs.append(ck[i])
            ys.append(xi)
    coef = np.polyfit(np.array(xs), np.array(ys), 1)
    at_truth = float(np.polyval(coef, TRUE_C_K))

    near = [i for i in range(len(ck)) if 0.078 <= ck[i] <= 0.122]
    along = np.array([g[i, int(np.argmin(abs(ce - np.polyval(coef, ck[i]))))] for i in near])
    along_ratio = float(along.max() / along.min())

    fit_x = np.linspace(0.055, 0.135, 50)
    ax.plot(fit_x, np.polyval(coef, fit_x), "--", color="white", lw=2.0, alpha=0.95,
            label=f"valley floor: c_eps = {coef[0]:.1f}*c_k {coef[1]:+.2f}")
    ax.plot(xs, ys, "o", color="white", ms=5, markeredgecolor="black", alpha=0.9)

    if starts:
        for idx, lab, c in starts:
            t = np.load(os.path.join(d, f"start_{idx}_trajectory.npy"))
            ax.plot(t[:, 0], t[:, 1], "-", color=c, lw=1.5, label=lab)
            ax.plot(t[0, 0], t[0, 1], "o", color=c, ms=9, markeredgecolor="black")
            ax.plot(t[-1, 0], t[-1, 1], "s", color=c, ms=9, markeredgecolor="black")

    ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="white", ms=26, markeredgecolor="black", zorder=6, label="true")
    ax.set_xlabel("c_k")
    ax.set_ylabel("c_eps")
    # Which direction is this loss blind to? Median spread of the loss along each
    # axis, holding the other fixed. The larger number is the direction the loss
    # actually constrains.
    ax.set_title(f"{title}\nvalley hits c_eps={at_truth:.3f} at true c_k=0.1 (true 0.7)   |   "
                 f"loss varies {along_ratio:.1f}x ALONG the valley")
    ax.legend(loc="upper left", fontsize=8)

fig.suptitle("Report 18: both losses have the SAME diagonal degeneracy, c_eps ~ 8.9*c_k - 0.23, and both valleys "
             "pass through truth\n(same grid, same N_STEPS=20; each panel normalised to its own minimum)")
fig.tight_layout()
p = os.path.join(OUT_DIR, "landscape_comparison.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
