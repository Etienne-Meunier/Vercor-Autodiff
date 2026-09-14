"""Report 21: observable error before and after assimilation, for one variant.

Re-runs the 10-day coupled rollout at four parameter settings -- the truth (which
defines the target), the initial guess, the lowest-loss iterate and the final
iterate -- and maps the error of that variant's own observable against the target.
Two figures are written, one for each choice of fitted point:

    error_before_after_best.png    lowest-loss iterate
    error_before_after.png         final iterate

Both are produced because the two disagree here. The loss constrains the
combination `c_eps - 8.9*c_k` far better than either parameter alone
([report-18](report-18-stratification-loss.md)), so a point can fit the field
better while sitting further from truth -- report-20's best-loss iterate had a
19.3% `c_k` error against the final iterate's 8.8%. Selecting on the loss uses no
knowledge of the true parameters and so is the choice available on a real
assimilation problem; selecting the closest-to-truth iterate is not. Quote one
consistently and never mix the best-loss field with the final-iterate parameters.

Variant selected exactly as in `fit.py`:

    REPORT21_OBS=mld_avg python snapshots.py

For the MLD observables the map domain is the intersection of the columns with a
well-defined MLD across all four rollouts, so every panel covers the same columns.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common21 import (
    AVG_WINDOW, INIT_C_EPS, INIT_C_K, N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K,
    build, make_observable,
)

OBS = os.environ.get("REPORT21_OBS", "mld_avg")
if OBS not in ("temp", "temp_avg", "mld", "mld_avg"):
    raise SystemExit(f"REPORT21_OBS must be one of temp/temp_avg/mld/mld_avg, got {OBS!r}")
AVERAGED = OBS.endswith("_avg")
IS_MLD = OBS.startswith("mld")

OUT_DIR = os.environ.get(
    "REPORT21_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", f"report-21-{OBS}")
)

UNIT = "m" if IS_MLD else "deg C"
QUANTITY = ("MLD" if IS_MLD else "surface temperature") + (f", mean over the last {AVG_WINDOW} days" if AVERAGED else "")

traj = np.load(os.path.join(OUT_DIR, "trajectory.npy"))
# Checkpoint row k+1 holds the parameters produced AFTER iteration k's update
# alongside the loss evaluated BEFORE it, i.e. at row k's parameters. Pairing
# naively names parameters one adam step past the ones that were scored.
losses = traj[1:, 2]
best_k = int(np.nanargmin(losses))
BEST = (float(traj[best_k, 0]), float(traj[best_k, 1]))
FINAL = (float(traj[-1, 0]), float(traj[-1, 1]))
print(f"observable={OBS}; {traj.shape[0] - 1} iterations", flush=True)
print(f"  best loss {np.nanmin(losses):.5e} at iteration {best_k}: c_k={BEST[0]:.4f}, c_eps={BEST[1]:.4f}", flush=True)
print(f"  final iterate: c_k={FINAL[0]:.4f}, c_eps={FINAL[1]:.4f}", flush=True)

couplers, initial_state, grid = build(segmented=AVERAGED)
observable = make_observable(OBS, couplers, initial_state, grid)
run = jax.jit(observable)
XT, YT = grid["longitude"], grid["latitude"]

fields, valids = {}, {}
for label, (a, b) in (("target", (TRUE_C_K, TRUE_C_EPS)),
                      ("init", (INIT_C_K, INIT_C_EPS)),
                      ("best", BEST),
                      ("final", FINAL)):
    t0 = time.time()
    f, v = run(jnp.asarray(a), jnp.asarray(b))
    fields[label], valids[label] = np.asarray(f), np.asarray(v)
    print(f"  {label:>6s} (c_k={a:.4f}, c_eps={b:.4f}): {time.time() - t0:.1f}s", flush=True)

# One domain for every panel, so the maps and the RMSEs are over the same cells.
DOMAIN = valids["target"] & valids["init"] & valids["best"] & valids["final"]
print(f"\nmap domain: {int(DOMAIN.sum())} cells "
      f"(target alone: {int(valids['target'].sum())})", flush=True)


def masked(a):
    return np.where(DOMAIN, a, np.nan)


def rmse(e):
    return float(np.sqrt(np.nanmean(e[~np.isnan(e)] ** 2)))


target = masked(fields["target"])
err = {k: masked(fields[k]) - target for k in ("init", "best", "final")}
r = {k: rmse(v) for k, v in err.items()}
print(f"{QUANTITY} RMSE: before {r['init']:.4f} {UNIT}; "
      f"best-loss iterate {r['best']:.4f} {UNIT} ({r['init'] / r['best']:.2f}x); "
      f"final iterate {r['final']:.4f} {UNIT} ({r['init'] / r['final']:.2f}x)", flush=True)
# Raw fields as well as the error fields: the paper figure must be redrawable
# (different colour limits, different panels, a difference of two fitted points)
# without another 4 rollouts on the cluster.
for k, v in err.items():
    np.save(os.path.join(OUT_DIR, f"err_{k}.npy"), v)
for k, v in fields.items():
    np.save(os.path.join(OUT_DIR, f"field_{k}.npy"), v)
np.save(os.path.join(OUT_DIR, "domain.npy"), DOMAIN)
np.save(os.path.join(OUT_DIR, "longitude.npy"), XT)
np.save(os.path.join(OUT_DIR, "latitude.npy"), YT)
np.save(os.path.join(OUT_DIR, "snapshot_params.npy"),
        np.array([[INIT_C_K, INIT_C_EPS], list(BEST), list(FINAL), [TRUE_C_K, TRUE_C_EPS]], float))


def figure(which, params, path):
    err_fit, r_fit = err[which], r[which]
    label = f"best loss, iter {best_k}" if which == "best" else f"final, iter {traj.shape[0] - 1}"

    # Robust colour limit: the error field is dominated by a handful of extreme
    # cells, so a 99th-percentile clip leaves the map almost entirely white. The
    # 95th brings out the structure; both panels share it so before/after compare.
    vmax = float(np.nanpercentile(np.abs(err["init"][~np.isnan(err["init"])]), 95))

    fig = plt.figure(figsize=(21, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.72], wspace=0.34)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

    im = axs[0].pcolormesh(XT, YT, target.T, cmap="viridis", shading="auto")
    axs[0].set_title(f"Target {QUANTITY}\ntrue c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}\n ", fontsize=11)
    fig.colorbar(im, ax=axs[0], label=UNIT, shrink=0.85)

    for ax, e, lab, (a, b), rv in (
        (axs[1], err["init"], "BEFORE assimilation", (INIT_C_K, INIT_C_EPS), r["init"]),
        (axs[2], err_fit, f"AFTER assimilation ({label})", params, r_fit),
    ):
        im = ax.pcolormesh(XT, YT, e.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"{lab}\nc_k={a:.4f}, c_eps={b:.4f}\nRMSE {rv:.4f} {UNIT}", fontsize=11)
    fig.colorbar(im, ax=[axs[1], axs[2]], label=f"{QUANTITY} error ({UNIT})", shrink=0.85, extend="both")

    ei = err["init"][~np.isnan(err["init"])]
    ef = err_fit[~np.isnan(err_fit)]
    lim = float(np.percentile(np.abs(ei), 99.5))
    bins = np.linspace(-lim, lim, 61)
    axs[3].hist(ei, bins=bins, color="tab:red", alpha=0.55, label=f"before (RMSE {r['init']:.4f} {UNIT})")
    axs[3].hist(ef, bins=bins, color="tab:green", alpha=0.6, label=f"after  (RMSE {r_fit:.4f} {UNIT})")
    axs[3].axvline(0, color="black", lw=0.8)
    axs[3].set_yscale("log")
    axs[3].set_xlabel(f"{QUANTITY} error ({UNIT})")
    axs[3].set_ylabel("cells")
    axs[3].set_title(f"Error distribution\nRMSE reduced {r['init'] / r_fit:.2f}x", fontsize=11)
    axs[3].legend(fontsize=8)
    axs[3].grid(alpha=0.3)

    for ax in axs[:3]:
        ax.set_xlabel("longitude")
    axs[0].set_ylabel("latitude")

    fig.suptitle(f"Report 21 ({OBS}): {QUANTITY} error before and after gradient-based assimilation "
                 f"-- {N_STEPS}-day coupled rollout, {traj.shape[0] - 1} adam iterations through the "
                 f"full OCN+LND+ATM model", fontsize=13, y=1.06)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


figure("best", BEST, os.path.join(OUT_DIR, "error_before_after_best.png"))
figure("final", FINAL, os.path.join(OUT_DIR, "error_before_after.png"))
