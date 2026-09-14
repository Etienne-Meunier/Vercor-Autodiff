"""Report 20: adam vs Gauss-Newton vs L-BFGS on the identical N=10 problem."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
OUT = os.path.join(FIG, "report-20-clip")
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

RUNS = [
    ("adam + clipping, 100 iters", "report-20-clip", "tab:green", "-o", 3),
    ("Gauss-Newton (forward-mode J)", "report-20-gn", "tab:blue", "-s", 8),
    ("L-BFGS (scipy)", "report-20-lbfgs", "tab:red", "-^", 7),
]

fig, axs = plt.subplots(1, 3, figsize=(20, 5.6))

for label, d, c, m, ms in RUNS:
    t = np.load(os.path.join(FIG, d, "trajectory.npy"))
    ck, ce, loss = t[:, 0], t[:, 1], t[:, 2]
    n = np.arange(len(t))
    axs[0].semilogy(n[1:], loss[1:], m, color=c, ms=ms, lw=1.5, label=label)
    axs[1].plot(ck, ce, m, color=c, ms=ms, lw=1.5, label=label, alpha=0.85)
    axs[1].plot(ck[-1], ce[-1], "*", color=c, ms=16, markeredgecolor="black", zorder=5)
    err = np.hypot((ck - TRUE_C_K) / TRUE_C_K, (ce - TRUE_C_EPS) / TRUE_C_EPS) * 100
    axs[2].semilogy(n, err, m, color=c, ms=ms, lw=1.5, label=label)

axs[0].set_xlabel("model evaluation / iteration")
axs[0].set_ylabel("loss")
axs[0].set_title("Loss")
axs[0].grid(alpha=0.3, which="both")
axs[0].legend(fontsize=8)

axs[1].plot(0.05, 0.40, "o", color="black", ms=11, label="start", zorder=6)
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=26, markeredgecolor="black",
            label="true", zorder=7)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter path\n(star = where each optimizer ended)")
# Zoomed to the region the runs actually work in. L-BFGS's cold-start step went
# to (1.0, 1e-4) -- both bounds -- which is off-scale here and would otherwise
# compress every other trajectory into a corner.
axs[1].set_xlim(0.03, 0.17)
axs[1].set_ylim(0.30, 0.75)
axs[1].annotate("L-BFGS first step goes\nto (1.0, 1e-4), off-scale",
                xy=(0.155, 0.335), fontsize=8, color="tab:red", ha="right")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)

axs[2].set_xlabel("model evaluation / iteration")
axs[2].set_ylabel("relative parameter error (%)")
axs[2].set_title("Distance from the true parameters\n(combined relative error in c_k and c_eps)")
axs[2].grid(alpha=0.3, which="both")
axs[2].legend(fontsize=8)

fig.suptitle("Report 20: three optimizers, identical 10-day coupled problem, identical loss and start -- "
             "Gauss-Newton wins because it builds curvature from one forward-mode Jacobian", fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, "optimizer_comparison.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
