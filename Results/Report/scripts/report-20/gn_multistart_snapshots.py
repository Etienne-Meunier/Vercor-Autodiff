"""Report 20b: surface-temperature error before and after Gauss-Newton, per start.

One rollout at each start's initial parameters and at the parameters Gauss-Newton
converged to, mapped against the target. All panels share one colour scale so the
starts are comparable with each other, not just within a column.
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

FIG = os.path.join(REPO_ROOT, "Results", "Report", "figures")
D = os.environ.get("REPORT20_OUT_DIR", os.path.join(FIG, "report-20-gn-multistart"))

STARTS = [("very-near", 0.095, 0.670), ("near", 0.090, 0.620), ("mid", 0.080, 0.550),
          ("report-15/20", 0.050, 0.400), ("far", 0.030, 0.300), ("above", 0.150, 0.850),
          ("off-diagonal", 0.130, 0.500)]

cpl, initial_state, grid = build()
SURF = grid["maskT"].shape[2] - 1
SMASK_NP = grid["maskT"][:, :, SURF]
XT, YT = grid["longitude"], grid["latitude"]


def surface_temp(a, b):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", jnp.asarray(a))
    p = set_veros_variable(p, "c_eps", jnp.asarray(b))
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    pay = cpl.run(st, output=None)._component_state("OCN").payload
    return pay.variables.temp[2:-2, 2:-2, SURF, pay.variables.tau]


run = jax.jit(surface_temp)
t0 = time.time()
TARGET = np.asarray(run(TRUE_C_K, TRUE_C_EPS))
print(f"target ({time.time() - t0:.0f}s)", flush=True)


def err_at(a, b):
    return np.where(SMASK_NP, np.asarray(run(a, b)) - TARGET, np.nan)


def rmse(e):
    return float(np.sqrt(np.nanmean(e[~np.isnan(e)] ** 2)))


def perr(a, b):
    return float(np.hypot((a - TRUE_C_K) / TRUE_C_K, (b - TRUE_C_EPS) / TRUE_C_EPS) * 100)


rows = []
for i, (lab, a0, b0) in enumerate(STARTS):
    tr = np.load(os.path.join(D, f"start_{i}_trajectory.npy"))
    a1, b1 = float(tr[-1, 0]), float(tr[-1, 1])
    e0, e1 = err_at(a0, b0), err_at(a1, b1)
    rows.append(dict(label=lab, a0=a0, b0=b0, a1=a1, b1=b1, e0=e0, e1=e1,
                     r0=rmse(e0), r1=rmse(e1), p0=perr(a0, b0), p1=perr(a1, b1)))
    print(f"{lab:>14s}: RMSE {rows[-1]['r0']:.4f} -> {rows[-1]['r1']:.4f} K "
          f"({rows[-1]['r0'] / rows[-1]['r1']:.2f}x)   param err {rows[-1]['p0']:.1f}% -> "
          f"{rows[-1]['p1']:.1f}%", flush=True)

allv = np.concatenate([r["e0"][~np.isnan(r["e0"])] for r in rows])
vmax = float(np.nanpercentile(np.abs(allv), 97))

n = len(rows)
fig, axs = plt.subplots(2, n, figsize=(3.1 * n, 8.0), sharex=True, sharey=True)
for j, r in enumerate(rows):
    for i, (key, tag, pk, pv, rm, pe) in enumerate((
            ("e0", "BEFORE", r["a0"], r["b0"], r["r0"], r["p0"]),
            ("e1", "AFTER", r["a1"], r["b1"], r["r1"], r["p1"]))):
        im = axs[i, j].pcolormesh(XT, YT, r[key].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        head = f"{r['label']}\n{tag}  ({pk:.4f}, {pv:.4f})" if i == 0 else f"{tag}  ({pk:.4f}, {pv:.4f})"
        axs[i, j].set_title(f"{head}\nparam err {pe:.1f}%   RMSE {rm:.4f} K", fontsize=8.5)
    axs[1, j].set_xlabel("longitude")
axs[0, 0].set_ylabel("latitude")
axs[1, 0].set_ylabel("latitude")
fig.colorbar(im, ax=axs.ravel().tolist(), label="surface temperature error (deg C)",
             shrink=0.6, extend="both", pad=0.012)
fig.suptitle(f"Report 20b: surface-temperature error before and after Gauss-Newton, per starting point "
             f"({N_STEPS}-day coupled rollout; shared colour scale)", fontsize=13, y=1.045)
p = os.path.join(D, "gn_multistart_snapshots.png")
fig.savefig(p, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}", flush=True)
np.save(os.path.join(D, "snapshot_rmse.npy"),
        np.array([[r["label"], r["p0"], r["p1"], r["r0"], r["r1"]] for r in rows], dtype=object),
        allow_pickle=True)
