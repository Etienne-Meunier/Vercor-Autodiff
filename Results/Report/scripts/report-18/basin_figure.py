"""Report 18: the loss has no basin around the true parameters -- it has a floor."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures", "report-18")

d, loss = np.load(os.path.join(OUT, "basin_width.npy")).T
LOSS_AT_TRUTH = 1.399912e-32
FIT_END = 2.9809e-05   # loss the 30-iteration fit finished at

fig, axs = plt.subplots(1, 2, figsize=(15, 5.8))

ax = axs[0]
ax.loglog(d, loss, "o-", color="crimson", ms=8, lw=2, label="measured loss")
# what a smooth quadratic basin through the widest point would look like
q = loss[0] * (d / d[0]) ** 2
ax.loglog(d, q, "--", color="gray", lw=2, label="smooth basin would give loss ~ d^2")
ax.axhline(FIT_END, color="tab:blue", ls=":", lw=2, label=f"where the 30-iter fit stopped ({FIT_END:.1e})")
ax.set_xlabel("d(c_eps): offset from the true c_eps = 0.7   (c_k held at 0.1)")
ax.set_ylabel("loss")
ax.set_title("Loss vs distance from truth\nflat over 3 decades: a noise floor, not a basin")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower right")
ax.annotate(f"loss AT truth = {LOSS_AT_TRUTH:.1e}\n(exact cancellation:\nbit-identical rollout)",
            xy=(d[-1], loss[-1]), xytext=(2e-5, 2.5e-5), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black"))

ax = axs[1]
ax.semilogx(d, loss / d ** 2, "o-", color="darkorange", ms=8, lw=2)
ax.set_yscale("log")
ax.set_xlabel("d(c_eps)")
ax.set_ylabel("loss / d^2")
ax.set_title("A smooth basin would make this constant.\nIt rises 7 decades instead.")
ax.grid(alpha=0.3, which="both")

fig.suptitle("Report 18: there is no basin of attraction around the true parameters -- "
             "chaotic decorrelation sets a loss floor of ~1e-5 at N_STEPS=20")
fig.tight_layout()
p = os.path.join(OUT, "basin_width.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
