"""Report 20: GN multistart paths, and the microscopic basin around the truth."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
D = os.path.join(FIG, "report-20-gn-multistart")
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

STARTS = [("very-near", 0.095, 0.670), ("near", 0.090, 0.620), ("mid", 0.080, 0.550),
          ("report-15/20", 0.050, 0.400), ("far", 0.030, 0.300), ("above", 0.150, 0.850),
          ("off-diagonal", 0.130, 0.500)]
COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(STARTS)))


def perr(a, b):
    return np.hypot((a - TRUE_C_K) / TRUE_C_K, (b - TRUE_C_EPS) / TRUE_C_EPS) * 100


fig, axs = plt.subplots(1, 3, figsize=(20, 5.8))

ax = axs[0]
for i, ((lab, a0, b0), c) in enumerate(zip(STARTS, COLORS)):
    t = np.load(os.path.join(D, f"start_{i}_trajectory.npy"))
    ax.plot(t[:, 0], t[:, 1], "-o", color=c, ms=4, lw=1.4, alpha=0.9,
            label=f"{lab}: {perr(a0, b0):.0f}% -> {perr(t[-1, 0], t[-1, 1]):.1f}%")
    ax.plot(a0, b0, "o", color=c, ms=10, markeredgecolor="black")
    ax.plot(t[-1, 0], t[-1, 1], "*", color=c, ms=17, markeredgecolor="black", zorder=5)
ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=30, markeredgecolor="black", zorder=6, label="true")
ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.set_title("Gauss-Newton from 7 starts\ncircle = start, star = end")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left")

ax = axs[1]
d0 = [perr(a, b) for _, a, b in STARTS]
d1, cols = [], []
for i, ((lab, a0, b0), c) in enumerate(zip(STARTS, COLORS)):
    t = np.load(os.path.join(D, f"start_{i}_trajectory.npy"))
    d1.append(perr(t[-1, 0], t[-1, 1]))
    cols.append(c)
ax.scatter(d0, d1, c=cols, s=160, edgecolors="black", zorder=3)
for (lab, _, _), x, y in zip(STARTS, d0, d1):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=8)
lim = max(max(d0), max(d1)) * 1.1
ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="no improvement")
ax.set_xlabel("parameter error at the start (%)")
ax.set_ylabel("parameter error at the end (%)")
ax.set_title("Final error vs starting error\nnot monotone: the two farthest starts do best")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axs[2]
z = np.load(os.path.join(D, "zoom_to_truth.npy"))
gap = np.abs(TRUE_C_K - z[:, 1])
loss = z[:, 3]
ok = gap > 0
ax.loglog(gap[ok], np.maximum(loss[ok], 1e-16), "-o", color="crimson", ms=9, lw=2, label="measured loss")
g = np.array([1e-8, 1e-5])
ax.loglog(g, loss[3] * (g / gap[3]) ** 2, "--", color="gray", lw=2, label="quadratic bowl (slope 2)")
ax.axvspan(1e-6, 1e-5, color="orange", alpha=0.18)
ax.text(3e-6, 1e-6, "transition", ha="center", fontsize=9, color="darkorange", rotation=90)
ax.set_xlabel("gap from the true c_k")
ax.set_ylabel("loss")
ax.set_title("Zoom on the truth\nsmooth basin only within ~1e-6 of c_k;\noutside it, a rugged plateau at ~1e-2")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Report 20: Gauss-Newton reaches the plateau that bounds this loss -- "
             "the smooth basin around the truth is only ~1e-6 wide", fontsize=13)
fig.tight_layout()
p = os.path.join(D, "gn_multistart.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
