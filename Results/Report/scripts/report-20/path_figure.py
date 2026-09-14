"""Report 20: loss along the straight line from the fitted point to the truth."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("REPORT20_OUT_DIR",
                     os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures", "report-20-clip"))
t, ck, ce, loss, gk, ge = np.load(os.path.join(OUT, "path_to_truth.npy")).T

fig, axs = plt.subplots(1, 2, figsize=(15, 5.4))

ax = axs[0]
ax.semilogy(t, np.maximum(loss, 1e-15), "-o", color="darkgreen", ms=7, lw=2)
ax.axhline(loss[0], color="tab:red", ls=":", lw=1.5, label=f"fitted point ({loss[0]:.2e})")
ax.set_xlabel("t: 0 = fitted parameters, 1 = true parameters")
ax.set_ylabel("loss")
ax.set_title("Loss along the straight path to the truth\n"
             "no barrier: it ends at 3e-14, and never rises above 1.43x the start")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="lower left")
ax.annotate("smooth quadratic bowl\nonly in the last ~15%", xy=(0.93, loss[-2]), xytext=(0.45, 1e-9),
            fontsize=9, arrowprops=dict(arrowstyle="->"))

ax = axs[1]
ax.plot(t, gk, "-o", color="tab:blue", ms=6, label="dL/dc_k")
ax.plot(t, ge, "-s", color="tab:red", ms=6, label="dL/dc_eps")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("t: 0 = fitted parameters, 1 = true parameters")
ax.set_ylabel("gradient")
ax.set_title("Gradient along the same path\n"
             "dL/dc_eps > 0 everywhere: descent says LOWER c_eps,\n"
             "while the truth lies at HIGHER c_eps")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("Report 20: why the fit stops short -- a narrow diagonal valley, not a local minimum",
             fontsize=13)
fig.tight_layout()
p = os.path.join(OUT, "path_to_truth.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
