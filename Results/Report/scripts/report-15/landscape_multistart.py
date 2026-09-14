"""Report 15 addendum: same 3 starting points as report-8's
landscape_multistart.py (c_k=0.05/c_eps=0.4, 0.15/0.85, 0.03/0.55),
re-optimized here with report-15's SURFACE-ONLY masked loss instead of
report-8's full-column unmasked one -- to see whether the cleaner loss
surface changes where the 3 runs land, not just how cleanly they get
there (report-15's single-start finding).

v2: adds the 12x12 forward-only grid scan (same c_k/c_eps ranges as
report-8's) as a contourf background for the parameter-trajectory panel
-- v1 skipped this pending a cost estimate. The 3 optimizations are
unchanged and resume instantly from their existing checkpoints.

Script: `Results/Report/scripts/report-15/landscape_multistart.py`.
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

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-15")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 20
N_ITERS = 30
LEARNING_RATE = 2e-2  # peak lr of the cosine-decay schedule
Z_LEVEL = -1  # surface level -- the loss target
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

# same grid range/resolution and 3 starts as report-8's landscape_multistart.py
CK_RANGE = (0.02, 0.18)
CEPS_RANGE = (0.25, 0.95)
GRID_N = 12

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

_surface_mask_np = np.asarray(
    initial_state._component_state("OCN").payload.variables.maskT[2:-2, 2:-2, Z_LEVEL]
).astype(bool)
surface_mask = jnp.asarray(_surface_mask_np)


def final_ocn_surface_temp(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    final_payload = result._component_state("OCN").payload
    tau = final_payload.variables.tau
    return final_payload.variables.temp[2:-2, 2:-2, Z_LEVEL, tau]


print(f"Generating target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...", flush=True)
target_surface_temp = jax.jit(final_ocn_surface_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_surface_temp = jax.lax.stop_gradient(target_surface_temp)

N_OCEAN_CELLS = int(_surface_mask_np.sum())
print(f"surface ocean cells in loss: {N_OCEAN_CELLS} / {_surface_mask_np.size} total surface cells", flush=True)


def loss_fn(params):
    temp = final_ocn_surface_temp(params["c_k"], params["c_eps"])
    diff = jnp.where(surface_mask, temp - target_surface_temp, 0.0)
    return jnp.sum(diff ** 2)


forward_loss_jit = jax.jit(lambda ck, ce: loss_fn({"c_k": ck, "c_eps": ce}))

GRID_PATH = os.path.join(OUT_DIR, "loss_grid.npy")
CK_GRID_PATH = os.path.join(OUT_DIR, "ck_grid.npy")
CEPS_GRID_PATH = os.path.join(OUT_DIR, "ceps_grid.npy")

ck_grid = np.linspace(*CK_RANGE, GRID_N)
ceps_grid = np.linspace(*CEPS_RANGE, GRID_N)

if os.path.exists(GRID_PATH) and os.path.exists(CK_GRID_PATH) and np.array_equal(np.load(CK_GRID_PATH), ck_grid):
    loss_grid = np.load(GRID_PATH)
    print(f"resuming grid scan from checkpoint ({np.count_nonzero(~np.isnan(loss_grid))} / {GRID_N * GRID_N} points already done)", flush=True)
else:
    loss_grid = np.full((GRID_N, GRID_N), np.nan)

print(f"\nGrid scan: {GRID_N}x{GRID_N} = {GRID_N * GRID_N} forward evals (surface-only masked loss) ...", flush=True)
for i, ck in enumerate(ck_grid):
    for j, ce in enumerate(ceps_grid):
        if not np.isnan(loss_grid[j, i]):
            continue
        loss_grid[j, i] = float(forward_loss_jit(jnp.asarray(ck), jnp.asarray(ce)))
        print(f"  c_k={ck:.4f}  c_eps={ce:.4f}  loss={loss_grid[j, i]:.4e}", flush=True)
        np.save(GRID_PATH, loss_grid)
        np.save(CK_GRID_PATH, ck_grid)
        np.save(CEPS_GRID_PATH, ceps_grid)


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

# --- figure: loss vs iteration + parameter-space trajectories (no grid backdrop) ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
colors = ["tab:orange", "tab:cyan", "tab:pink"]

for traj, color in zip(all_trajectories, colors):
    losses = [t[2] for t in traj[1:]]
    axs[0].plot(range(len(losses)), losses, "-o", color=color, markersize=3, label=f"start ({traj[0][0]:.2f}, {traj[0][1]:.2f})")
axs[0].set_yscale("log")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (surface, masked)")
axs[0].set_title("Loss vs iteration")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

cf = axs[1].pcolormesh(ck_grid, ceps_grid, loss_grid, cmap="turbo", norm=plt.matplotlib.colors.LogNorm(), shading="nearest")
fig.colorbar(cf, ax=axs[1], label="loss (surface, masked)", shrink=0.85)

for traj, color in zip(all_trajectories, colors):
    cks = [t[0] for t in traj]
    ceps = [t[1] for t in traj]
    axs[1].plot(cks, ceps, "-o", color=color, markersize=3, linewidth=1.5, label=f"start ({cks[0]:.2f}, {ceps[0]:.2f})")
    axs[1].plot(cks[0], ceps[0], "o", color=color, markersize=9, markeredgecolor="black", zorder=5)
    axs[1].plot(cks[-1], ceps[-1], "s", color=color, markersize=9, markeredgecolor="black", zorder=5)
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=22, markeredgecolor="black", label="true", zorder=6)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectories + loss landscape")
axs[1].legend(fontsize=8, loc="upper left")

fig.suptitle(f"Report 15: 3-start multistart, SURFACE-ONLY masked loss, N_STEPS={N_STEPS}")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, "landscape_multistart.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nsaved {out_path}", flush=True)

print("\n=== summary ===", flush=True)
for traj, (sck, sce) in zip(all_trajectories, STARTS):
    iters = traj[1:]
    losses = np.array([t[2] for t in iters])
    best_i = int(np.argmin(losses))
    best_ck, best_ceps, best_loss = iters[best_i]
    end_ck, end_ceps, end_loss = iters[-1]
    print(f"start ({sck}, {sce}): best iter={best_i} loss={best_loss:.4f} (c_k={best_ck:.4f}, c_eps={best_ceps:.4f})  "
          f"end (c_k={end_ck:.4f}, c_eps={end_ceps:.4f}) loss={end_loss:.4f}", flush=True)
