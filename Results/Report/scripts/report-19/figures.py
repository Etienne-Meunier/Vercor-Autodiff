"""Report 19 figures: where the chaos floor switches on, and what survives it."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures", "report-19")

rows = np.load(os.path.join(OUT, "signal_to_floor.npy"), allow_pickle=True)
Ns = sorted({int(r[0]) for r in rows})
obs = ["temp_surf", "temp_surf_tmean", "dTdz_absavg3", "dTdz_absavg3_tmean", "mld", "mld_tmean"]
COL = {"temp_surf": "tab:blue", "temp_surf_tmean": "tab:cyan",
       "dTdz_absavg3": "tab:red", "dTdz_absavg3_tmean": "tab:orange",
       "mld": "tab:green", "mld_tmean": "limegreen"}


def series(name, idx):
    return [next(float(r[idx]) for r in rows if int(r[0]) == n and r[1] == name) for n in Ns]


fig, axs = plt.subplots(1, 3, figsize=(20, 6))

ax = axs[0]
for o in obs:
    ax.semilogy(Ns, series(o, 3), "-o" if "tmean" not in o else "--s", color=COL[o], ms=7, label=o)
ax.axvspan(10, 15, color="red", alpha=0.10)
ax.text(12.5, ax.get_ylim()[0], " chaos\n onset", ha="center", va="bottom", fontsize=9, color="darkred")
ax.set_xlabel("rollout length (days)")
ax.set_ylabel("floor: loss at a 1e-5 c_eps perturbation")
ax.set_title("The decorrelation floor switches on\nbetween day 10 and day 15")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axs[1]
for o in obs:
    ax.semilogy(Ns, series(o, 6), "-o" if "tmean" not in o else "--s", color=COL[o], ms=7, label=o)
ax.axhline(1.0, color="black", lw=1)
ax.axvspan(10, 15, color="red", alpha=0.10)
ax.set_xlabel("rollout length (days)")
ax.set_ylabel("signal / floor  for a 30% c_eps error")
ax.set_title("Usable dynamic range for c_eps\n(higher is better; 1 = indistinguishable from noise)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axs[2]
long_Ns = [n for n in Ns if n >= 15]
x = np.arange(len(long_Ns))
w = 0.13
for i, o in enumerate(obs):
    vals = [next(float(r[6]) for r in rows if int(r[0]) == n and r[1] == o) for n in long_Ns]
    ax.bar(x + (i - 2.5) * w, vals, w, color=COL[o], label=o)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in long_Ns])
ax.set_ylabel("signal / floor for a 30% c_eps error")
ax.set_title("Past the onset, time-averaging is what survives\n(dashed = time-averaged variants)")
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=8)

fig.suptitle("Report 19: the chaos floor, not the observable, is what limits a 20-day fit "
             "-- and it does not exist before day ~12")
fig.tight_layout()
p = os.path.join(OUT, "signal_to_floor.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}")
