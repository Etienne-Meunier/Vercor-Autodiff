"""Report 8 addendum: 2D loss-landscape grid scan (forward-only, no grad) at
N_STEPS=20, as background contour for the (c_k, c_eps) parameter-space
plot, with GD trajectories from 3 starting points overlaid: the existing
report-8 run (c_k=0.05, c_eps=0.4, hardcoded from that run's log) plus two
new optimizations run here.

HOLD: do not launch until told to. ~2h total: grid scan (~36 forward
evals) + 2 new 20-iteration optimizations.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veros_gcm,
)
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-8")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 20
N_ITERS = 20
LEARNING_RATE = 2e-2
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

# grid covers all three starting points and the true value
CK_RANGE = (0.02, 0.18)
CEPS_RANGE = (0.25, 0.95)
GRID_N = 6

# report-8's already-completed run (c_k=0.05, c_eps=0.4 start), hardcoded
# from Results/Report/scripts/report-8/fit_and_generate_figures.py's log
REPORT8_TRAJECTORY = [
    (0.0500, 0.4000, None),
    (0.0700, 0.3800, 4.629689e+01),
    (0.0872, 0.3648, 1.874484e+01),
    (0.1011, 0.3614, 4.778722e+00),
    (0.1122, 0.3680, 2.812467e+00),
    (0.1211, 0.3795, 6.913485e+00),
    (0.1284, 0.3936, 1.125722e+01),
    (0.1339, 0.4093, 1.646632e+01),
    (0.1380, 0.4262, 2.064083e+01),
    (0.1395, 0.4438, 2.364784e+01),
    (0.1405, 0.4620, 2.424678e+01),
    (0.1407, 0.4804, 2.481679e+01),
    (0.1400, 0.4990, 2.446280e+01),
    (0.1317, 0.5177, 2.377855e+01),
    (0.1237, 0.5361, 1.666441e+01),
    (0.1163, 0.5538, 1.021156e+01),
    (0.1091, 0.5707, 7.347647e+00),
    (0.1022, 0.5865, 2.888215e+00),
    (0.0959, 0.6011, 8.773194e-01),
    (0.0905, 0.6145, 9.867251e-01),
    (0.0863, 0.6261, 2.747287e+00),
]

# two new starting points: opposite corner, and a third off-diagonal point
EXTRA_STARTS = [
    (0.15, 0.85),
    (0.03, 0.55),
]

ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg_learning",
        uses_atmosphere_forcing=True,
        restore_to_climatology=True,
        jitted=True,
        execution="jax",
    ),
)
jcm_setup = make_jcm_land_atmosphere(
    ocn.grid,
    config=JCMLandAtmosphereConfig(
        atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True),
    ),
)
lnd = jcm_setup.land
atm = jcm_setup.atmosphere

clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=N_STEPS, calendar="noleap")
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)
cpl = Coupler(
    clock=clock,
    components=[ocn, lnd, atm],
    exchanges=exchanges,
    run_order=["OCN", "LND", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)
initial_state = cpl.initial_state()


def final_ocn_temp(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload.variables.temp


print(f"Generating target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...")
target_temp = jax.jit(final_ocn_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_temp = jax.lax.stop_gradient(target_temp)


def loss_fn(params):
    temp = final_ocn_temp(params["c_k"], params["c_eps"])
    return jnp.sum((temp - target_temp) ** 2)


forward_loss_jit = jax.jit(lambda ck, ce: loss_fn({"c_k": ck, "c_eps": ce}))

GRID_PATH = os.path.join(OUT_DIR, "loss_grid.npy")
CK_GRID_PATH = os.path.join(OUT_DIR, "ck_grid.npy")
CEPS_GRID_PATH = os.path.join(OUT_DIR, "ceps_grid.npy")

# --- grid scan (forward-only), resumable + incrementally checkpointed ---
ck_grid = np.linspace(*CK_RANGE, GRID_N)
ceps_grid = np.linspace(*CEPS_RANGE, GRID_N)

if os.path.exists(GRID_PATH) and os.path.exists(CK_GRID_PATH) and np.array_equal(np.load(CK_GRID_PATH), ck_grid):
    loss_grid = np.load(GRID_PATH)
    print(f"resuming grid scan from checkpoint ({np.count_nonzero(loss_grid)} / {GRID_N * GRID_N} points already done)")
else:
    loss_grid = np.full((GRID_N, GRID_N), np.nan)

print(f"\nGrid scan: {GRID_N}x{GRID_N} = {GRID_N * GRID_N} forward evals ...", flush=True)
for i, ck in enumerate(ck_grid):
    for j, ce in enumerate(ceps_grid):
        if not np.isnan(loss_grid[j, i]):
            continue
        loss_grid[j, i] = float(forward_loss_jit(jnp.asarray(ck), jnp.asarray(ce)))
        print(f"  c_k={ck:.4f}  c_eps={ce:.4f}  loss={loss_grid[j, i]:.4e}", flush=True)
        np.save(GRID_PATH, loss_grid)
        np.save(CK_GRID_PATH, ck_grid)
        np.save(CEPS_GRID_PATH, ceps_grid)


# --- new optimization runs, resumable + checkpointed per start ---
def run_optimization(init_ck, init_ceps, checkpoint_path):
    if os.path.exists(checkpoint_path):
        traj = [tuple(row) for row in np.load(checkpoint_path).tolist()]
        print(f"resuming optimization from checkpoint ({len(traj) - 1} / {N_ITERS} iters already done)", flush=True)
    else:
        traj = [(init_ck, init_ceps, None)]

    params = {"c_k": jnp.asarray(traj[-1][0]), "c_eps": jnp.asarray(traj[-1][1])}
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    for iteration in range(len(traj) - 1, N_ITERS):
        loss_value, grads = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
        traj.append((ck_val, ceps_val, float(loss_value)))
        print(f"  iter {iteration:3d}  loss={float(loss_value):.4e}  c_k={ck_val:.4f}  c_eps={ceps_val:.4f}", flush=True)
        np.save(checkpoint_path, np.array([[t[0], t[1], t[2] if t[2] is not None else np.nan] for t in traj]))
    return traj


all_trajectories = [REPORT8_TRAJECTORY]
for idx, (start_ck, start_ceps) in enumerate(EXTRA_STARTS):
    print(f"\nOptimizing from start (c_k={start_ck}, c_eps={start_ceps}) ...", flush=True)
    ckpt_path = os.path.join(OUT_DIR, f"extra_start_{idx}_trajectory.npy")
    all_trajectories.append(run_optimization(start_ck, start_ceps, ckpt_path))

# --- figure: loss landscape + trajectories ---
fig, ax = plt.subplots(figsize=(8, 7))
cf = ax.contourf(ck_grid, ceps_grid, loss_grid, levels=30, cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
fig.colorbar(cf, ax=ax, label="loss (sum squared temp error)")

colors = ["tab:orange", "tab:cyan", "tab:pink"]
for traj, color in zip(all_trajectories, colors):
    cks = [t[0] for t in traj]
    ceps = [t[1] for t in traj]
    ax.plot(cks, ceps, "-o", color=color, markersize=3, linewidth=1.5, label=f"start ({cks[0]:.2f}, {ceps[0]:.2f})")
    ax.plot(cks[0], ceps[0], "o", color=color, markersize=9, markeredgecolor="black", zorder=5)
    ax.plot(cks[-1], ceps[-1], "s", color=color, markersize=9, markeredgecolor="black", zorder=5)

ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=22, markeredgecolor="black", label="true", zorder=6)
ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.set_title(f"Report 8: loss landscape + GD trajectories, N_STEPS={N_STEPS}")
ax.legend(loc="upper left", fontsize=8)

landscape_path = os.path.join(OUT_DIR, "landscape_multistart.png")
fig.savefig(landscape_path, dpi=150)
plt.close(fig)
print(f"\nsaved {landscape_path}")
