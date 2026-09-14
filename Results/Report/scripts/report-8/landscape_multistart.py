"""Report 8 addendum: 2D loss-landscape grid scan (forward-only, no grad) at
N_STEPS=20, as background contour for the (c_k, c_eps) parameter-space
plot, with GD trajectories from 3 starting points overlaid.

v2 (this file): grid finer 6x6 -> 12x12 (144 forward evals). All three
starts (including the original c_k=0.05/c_eps=0.4 one, previously just
hardcoded from the old fixed-lr run) now re-optimized fresh with the
cosine-decay lr schedule + N_ITERS=30 recipe from
fit_and_generate_figures.py v2, instead of the old fixed-lr/20-iter one.
Checkpoints renamed (start_{i}_trajectory.npy) since the optimizer state
is not compatible with the old extra_start_*.npy files from the v1 run.

Meant for remote/GPU launch (grid + 3x30-iter cosine optimizations is
several hours of coupled-model forward+backward passes).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

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
N_ITERS = 30  # matches fit_and_generate_figures.py v2's cosine-decay recipe
LEARNING_RATE = 2e-2  # peak lr of the cosine-decay schedule
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

# grid covers all three starting points and the true value
CK_RANGE = (0.02, 0.18)
CEPS_RANGE = (0.25, 0.95)
GRID_N = 12  # finer than v1's 6x6

# three starting points: the original report-8 start, opposite corner,
# and a third off-diagonal point -- all re-run fresh with the cosine
# schedule (v1 reused a hardcoded fixed-lr trajectory for the first one)
STARTS = [
    (0.05, 0.40),
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
    schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
    optimizer = optax.adam(schedule)
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


all_trajectories = []
for idx, (start_ck, start_ceps) in enumerate(STARTS):
    print(f"\nOptimizing from start (c_k={start_ck}, c_eps={start_ceps}) ...", flush=True)
    ckpt_path = os.path.join(OUT_DIR, f"start_{idx}_trajectory.npy")
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
