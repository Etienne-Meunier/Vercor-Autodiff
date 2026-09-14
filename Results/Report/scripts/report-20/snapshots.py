"""Report 20: surface-temperature error before and after assimilation.

Re-runs the N=10 coupled rollout at three parameter settings -- the truth (which
defines the target), the initial guess, and the fitted parameters -- and maps the
surface temperature error against the target for the initial and fitted cases.

Reads the fitted parameters from the run's trajectory.npy, so it can be pointed
at any of the report-20 output directories:

    REPORT20_OUT_DIR=.../figures/report-20-clip python snapshots.py
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "100")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import (  # noqa: E402
    INIT_C_EPS, INIT_C_K, N_ITERS, N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K,
    best_iterate, build,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20-clip"))

traj = [tuple(r) for r in np.load(os.path.join(OUT_DIR, "trajectory.npy")).tolist()]
best_iter, best_loss, best_ck, best_ceps = best_iterate(traj)
# REPORT20_USE_BEST=1 snapshots the lowest-loss iterate instead of the last one.
# Note these can disagree about which is "better": the loss and the parameter
# error are not aligned here, because the loss constrains the combination
# c_eps - 8.9*c_k far better than either parameter alone (report-18). Quote
# whichever is used consistently -- do not mix the best-loss field with the
# final-iterate parameter errors.
USE_BEST = os.environ.get("REPORT20_USE_BEST", "") not in ("", "0")
if USE_BEST:
    FIT_C_K, FIT_C_EPS, WHICH = best_ck, best_ceps, f"best loss, iter {best_iter}"
else:
    FIT_C_K, FIT_C_EPS, WHICH = traj[-1][0], traj[-1][1], f"final, iter {len(traj) - 1}"
print(f"{len(traj) - 1} iterations; final c_k={traj[-1][0]:.4f}, c_eps={traj[-1][1]:.4f}; "
      f"best (iter {best_iter}, loss {best_loss:.4e}) c_k={best_ck:.4f}, c_eps={best_ceps:.4f}", flush=True)
print(f"snapshotting: {WHICH} -> c_k={FIT_C_K:.4f}, c_eps={FIT_C_EPS:.4f}", flush=True)

cpl, initial_state, grid = build()
MASK = grid["maskT"]
SURF = MASK.shape[2] - 1
SMASK = MASK[:, :, SURF]
XT, YT = grid["longitude"], grid["latitude"]


def surface_temp(c_k, c_eps):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", jnp.asarray(c_k))
    p = set_veros_variable(p, "c_eps", jnp.asarray(c_eps))
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    pay = cpl.run(st, output=None)._component_state("OCN").payload
    return pay.variables.temp[2:-2, 2:-2, SURF, pay.variables.tau]


run = jax.jit(surface_temp)
fields = {}
for label, (a, b) in (("target", (TRUE_C_K, TRUE_C_EPS)),
                      ("init", (INIT_C_K, INIT_C_EPS)),
                      ("fitted", (FIT_C_K, FIT_C_EPS))):
    t0 = time.time()
    fields[label] = np.asarray(run(a, b))
    print(f"  {label:>7s} (c_k={a:.4f}, c_eps={b:.4f}): {time.time() - t0:.1f}s", flush=True)


def masked(a):
    return np.where(SMASK, a, np.nan)


target = masked(fields["target"])
err_init = masked(fields["init"]) - target
err_fit = masked(fields["fitted"]) - target


def rmse(e):
    return float(np.sqrt(np.nanmean(e[~np.isnan(e)] ** 2)))


r_i, r_f = rmse(err_init), rmse(err_fit)
print(f"\nsurface temperature RMSE  before {r_i:.4f} K   after {r_f:.4f} K   "
      f"reduction {r_i / r_f:.1f}x", flush=True)
np.save(os.path.join(OUT_DIR, "err_init.npy"), err_init)
np.save(os.path.join(OUT_DIR, "err_fitted_best.npy" if USE_BEST else "err_fitted.npy"), err_fit)

# Robust colour limit: the error field is dominated by a handful of extreme cells,
# so a 99th-percentile clip leaves the map almost entirely white. The 95th brings
# out the structure; both panels share it so before/after are comparable.
vmax = float(np.nanpercentile(np.abs(err_init[~np.isnan(err_init)]), 95))

fig = plt.figure(figsize=(21, 5.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.72], wspace=0.34)
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

im = axs[0].pcolormesh(XT, YT, target.T, cmap="viridis", shading="auto")
axs[0].set_title(f"Target surface temperature\ntrue c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}\n ", fontsize=11)
fig.colorbar(im, ax=axs[0], label="deg C", shrink=0.85)

for ax, err, lab, (a, b), r in (
    (axs[1], err_init, "BEFORE assimilation", (INIT_C_K, INIT_C_EPS), r_i),
    (axs[2], err_fit, f"AFTER assimilation ({WHICH})", (FIT_C_K, FIT_C_EPS), r_f),
):
    im = ax.pcolormesh(XT, YT, err.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"{lab}\nc_k={a:.4f}, c_eps={b:.4f}\nRMSE {r:.4f} K", fontsize=11)
fig.colorbar(im, ax=[axs[1], axs[2]], label="surface temperature error (deg C)",
             shrink=0.85, extend="both")

# Error distribution: the clearest single view of the improvement.
ei = err_init[~np.isnan(err_init)]
ef = err_fit[~np.isnan(err_fit)]
lim = float(np.percentile(np.abs(ei), 99.5))
bins = np.linspace(-lim, lim, 61)
axs[3].hist(ei, bins=bins, color="tab:red", alpha=0.55, label=f"before (RMSE {r_i:.4f} K)")
axs[3].hist(ef, bins=bins, color="tab:green", alpha=0.6, label=f"after  (RMSE {r_f:.4f} K)")
axs[3].axvline(0, color="black", lw=0.8)
axs[3].set_yscale("log")
axs[3].set_xlabel("surface temperature error (deg C)")
axs[3].set_ylabel("ocean cells")
axs[3].set_title(f"Error distribution\nRMSE reduced {r_i / r_f:.1f}x", fontsize=11)
axs[3].legend(fontsize=8)
axs[3].grid(alpha=0.3)

for ax in axs[:3]:
    ax.set_xlabel("longitude")
axs[0].set_ylabel("latitude")

fig.suptitle(f"Report 20: surface temperature error before and after gradient-based assimilation "
             f"-- {N_STEPS}-day coupled rollout, {len(traj) - 1} adam iterations through the full "
             f"OCN+LND+ATM model", fontsize=13, y=1.06)
p = os.path.join(OUT_DIR, "error_before_after_best.png" if USE_BEST else "error_before_after.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}", flush=True)
