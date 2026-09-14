"""Report 20: mixed-layer-depth error before and after the surface-temperature fit.

Companion to snapshots.py. Same three parameter settings, same layout -- but the
diagnostic mapped here is MLD, which was NOT part of the loss. The surface
temperature fit never saw it, so this asks whether correcting the parameters
against temperature also corrects the mixed layer the same parameters control.

MLD kernel is report-16's: depth where potential density first exceeds
prho(reference level) + 0.03 kg/m^3, linearly interpolated between the two levels
bracketing the crossing. Columns without a well-defined MLD in any of the three
runs are excluded, so all panels share one mask.
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "100")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import (  # noqa: E402
    INIT_C_EPS, INIT_C_K, N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, best_iterate, build,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20-clip"))
MLD_REF_DEPTH, MLD_REF_OFFSET = -10.0, 0.03

traj = [tuple(r) for r in np.load(os.path.join(OUT_DIR, "trajectory.npy")).tolist()]
best_iter, best_loss, best_ck, best_ceps = best_iterate(traj)
USE_BEST = os.environ.get("REPORT20_USE_BEST", "") not in ("", "0")
if USE_BEST:
    FIT_C_K, FIT_C_EPS, WHICH = best_ck, best_ceps, f"best loss, iter {best_iter}"
else:
    FIT_C_K, FIT_C_EPS, WHICH = traj[-1][0], traj[-1][1], f"final, iter {len(traj) - 1}"
print(f"snapshotting MLD at: {WHICH} -> c_k={FIT_C_K:.4f}, c_eps={FIT_C_EPS:.4f}", flush=True)

cpl, initial_state, grid = build()
MASKT = jnp.asarray(grid["maskT"])
ZT = jnp.asarray(grid["zt"])
XT, YT = grid["longitude"], grid["latitude"]


def mld_of(c_k, c_eps):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", jnp.asarray(c_k))
    p = set_veros_variable(p, "c_eps", jnp.asarray(c_eps))
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    prho = cpl.run(st, output=None)._component_state("OCN").payload.variables.prho[2:-2, 2:-2, :]

    level = jnp.arange(ZT.shape[-1])
    ridx = jnp.max(jnp.where(ZT < MLD_REF_DEPTH, level, -1))
    prho_ref = prho[:, :, ridx] + MLD_REF_OFFSET
    valid = MASKT.astype(bool) & (level <= ridx)
    drho = prho - prho_ref[:, :, jnp.newaxis]
    below = valid & (drho > 0)
    i_below = jnp.argmax(jnp.where(below, ZT, -jnp.inf), axis=-1)
    depth_below = ZT[i_below]
    above = valid & (drho < 0) & (ZT > depth_below[:, :, jnp.newaxis])
    i_above = jnp.argmin(jnp.where(above, ZT, jnp.inf), axis=-1)
    wd = jnp.any(below, axis=-1) & jnp.any(above, axis=-1)

    pb = jnp.take_along_axis(prho, i_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    pa = jnp.take_along_axis(prho, i_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    denom = jnp.where(wd, pa - pb, 1.0)
    mld = (prho_ref - pb) / denom * (ZT[i_above] - depth_below) + depth_below
    return jnp.where(wd, mld, 0.0), wd


run = jax.jit(mld_of)
fields, masks = {}, []
for label, (a, b) in (("target", (TRUE_C_K, TRUE_C_EPS)),
                      ("init", (INIT_C_K, INIT_C_EPS)),
                      ("fitted", (FIT_C_K, FIT_C_EPS))):
    t0 = time.time()
    m, wd = run(a, b)
    fields[label] = np.asarray(m)
    masks.append(np.asarray(wd))
    print(f"  {label:>7s} (c_k={a:.4f}, c_eps={b:.4f}): {time.time() - t0:.1f}s", flush=True)

MASK = masks[0] & masks[1] & masks[2]
print(f"columns with a well-defined MLD in all three runs: {int(MASK.sum())}", flush=True)


def masked(a):
    return np.where(MASK, a, np.nan)


target = masked(fields["target"])
err_init = masked(fields["init"]) - target
err_fit = masked(fields["fitted"]) - target


def rmse(e):
    return float(np.sqrt(np.nanmean(e[~np.isnan(e)] ** 2)))


r_i, r_f = rmse(err_init), rmse(err_fit)
print(f"\nMLD RMSE  before {r_i:.4f} m   after {r_f:.4f} m   change {r_i / r_f:.2f}x", flush=True)
np.save(os.path.join(OUT_DIR, "mld_err_init.npy"), err_init)
np.save(os.path.join(OUT_DIR, "mld_err_fitted.npy"), err_fit)

vmax = float(np.nanpercentile(np.abs(err_init[~np.isnan(err_init)]), 95))
fig = plt.figure(figsize=(21, 5.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.72], wspace=0.34)
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

im = axs[0].pcolormesh(XT, YT, target.T, cmap="viridis", shading="auto")
axs[0].set_title(f"Target mixed layer depth\ntrue c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}\n ", fontsize=11)
fig.colorbar(im, ax=axs[0], label="MLD (m)", shrink=0.85)

for ax, err, lab, (a, b), r in (
    (axs[1], err_init, "BEFORE assimilation", (INIT_C_K, INIT_C_EPS), r_i),
    (axs[2], err_fit, f"AFTER assimilation ({WHICH})", (FIT_C_K, FIT_C_EPS), r_f),
):
    im = ax.pcolormesh(XT, YT, err.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"{lab}\nc_k={a:.4f}, c_eps={b:.4f}\nRMSE {r:.4f} m", fontsize=11)
fig.colorbar(im, ax=[axs[1], axs[2]], label="MLD error (m)", shrink=0.85, extend="both")

ei = err_init[~np.isnan(err_init)]
ef = err_fit[~np.isnan(err_fit)]
lim = float(np.percentile(np.abs(ei), 99.5))
bins = np.linspace(-lim, lim, 61)
axs[3].hist(ei, bins=bins, color="tab:red", alpha=0.55, label=f"before (RMSE {r_i:.4f} m)")
axs[3].hist(ef, bins=bins, color="tab:green", alpha=0.6, label=f"after  (RMSE {r_f:.4f} m)")
axs[3].axvline(0, color="black", lw=0.8)
axs[3].set_yscale("log")
axs[3].set_xlabel("MLD error (m)")
axs[3].set_ylabel("ocean columns")
axs[3].set_title(f"Error distribution\nRMSE {'reduced' if r_f < r_i else 'INCREASED'} {r_i / r_f:.2f}x", fontsize=11)
axs[3].legend(fontsize=8)
axs[3].grid(alpha=0.3)

for ax in axs[:3]:
    ax.set_xlabel("longitude")
axs[0].set_ylabel("latitude")

fig.suptitle(f"Report 20: mixed-layer-depth error before and after the SURFACE-TEMPERATURE fit "
             f"-- MLD was not in the loss ({N_STEPS}-day coupled rollout)", fontsize=13, y=1.06)
p = os.path.join(OUT_DIR, "mld_error_before_after" + ("_best" if USE_BEST else "") + ".png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}", flush=True)
