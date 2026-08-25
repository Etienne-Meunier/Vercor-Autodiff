"""Report 2: c_k/c_eps fit over a longer coupled rollout (N_STEPS=10, 2x
report-1's 5 steps) than report-1, same coupled OCN+LND+ATM(jcm) model.
Runs the fit itself (unlike report-1's figure script, which reused an
already-completed fit's numbers) and produces the same two figures
(trajectory, temperature bias snapshot), following the same pattern as
Veros-Autodiff's report-2/section3b_ck_ceps_long_rollout.py.

No 2D loss-landscape grid scan here (unlike that reference script) -- each
grid point needs a full forward rollout, and even a coarse 5x5 grid would
cost as much as the whole fit; out of scope for this pass.
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

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-2")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 10  # 2x report-1's 5 steps
N_ITERS = 10
LEARNING_RATE = 2e-2

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4
Z_LEVEL = -1  # surface level

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


def final_ocn_payload(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload


def final_ocn_temp(c_k, c_eps):
    return final_ocn_payload(c_k, c_eps).variables.temp


print(f"Generating synthetic target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...")
target_temp = jax.jit(final_ocn_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_temp = jax.lax.stop_gradient(target_temp)


def loss_fn(params):
    temp = final_ocn_temp(params["c_k"], params["c_eps"])
    return jnp.sum((temp - target_temp) ** 2)


params = {"c_k": jnp.asarray(INIT_C_K), "c_eps": jnp.asarray(INIT_C_EPS)}
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)
value_and_grad_fn = jax.value_and_grad(loss_fn)

trajectory = [(INIT_C_K, INIT_C_EPS, None)]
print(f"\n{'iter':>4s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}")
print(f"{'init':>4s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")
for iteration in range(N_ITERS):
    loss_value, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    trajectory.append((ck_val, ceps_val, float(loss_value)))
    print(f"{iteration:4d}  {float(loss_value):14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}")

FINAL_C_K, FINAL_C_EPS = trajectory[-1][0], trajectory[-1][1]
print("\nrecovered c_k:  ", FINAL_C_K, " (true:", TRUE_C_K, ")")
print("recovered c_eps:", FINAL_C_EPS, " (true:", TRUE_C_EPS, ")")

# --- snapshot rollouts (target already have; initial + optimized) ---
print("\nrolling out initial guess for snapshot ...")
initial_payload = jax.jit(final_ocn_payload)(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS))
print("rolling out optimized for snapshot ...")
optimized_payload = jax.jit(final_ocn_payload)(jnp.asarray(FINAL_C_K), jnp.asarray(FINAL_C_EPS))
target_payload_full = jax.jit(final_ocn_payload)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))


def field_at_level(payload, z):
    tau = payload.variables.tau
    temp = np.asarray(payload.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(payload.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


xt = np.asarray(ocn.grid.longitude)
yt = np.asarray(ocn.grid.latitude)

target_field = field_at_level(target_payload_full, Z_LEVEL)
bias_initial = field_at_level(initial_payload, Z_LEVEL) - target_field
bias_optimized = field_at_level(optimized_payload, Z_LEVEL) - target_field

bias_max = np.nanmax(
    np.abs(
        np.concatenate(
            [bias_initial[~np.isnan(bias_initial)], bias_optimized[~np.isnan(bias_optimized)]]
        )
    )
)

# --- figure 1: trajectory ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

iters = list(range(len(trajectory) - 1))
losses = [t[2] for t in trajectory[1:]]
axs[0].plot(iters, losses, "o-", color="tab:orange")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared temp error)")
axs[0].set_title("Loss vs iteration")
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="tab:orange", markersize=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:orange", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 2: c_k/c_eps recovery, {N_STEPS}-step coupled OCN+LND+ATM(jcm) rollout")
fig.tight_layout()
traj_path = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"saved {traj_path}")

# --- figure 2: temperature bias snapshot ---
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[0].set_title(f"target (surface, c_k={TRUE_C_K}, c_eps={TRUE_C_EPS})")
fig.colorbar(im0, ax=axs[0], label="temp", shrink=0.8)

im1 = axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title(f"initial - target (c_k={INIT_C_K}, c_eps={INIT_C_EPS})")

im2 = axs[2].pcolormesh(xt, yt, bias_optimized.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title(f"optimized - target (c_k={FINAL_C_K:.4f}, c_eps={FINAL_C_EPS:.4f})")
fig.colorbar(im2, ax=[axs[1], axs[2]], label="bias", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"Report 2: temperature bias after {N_STEPS} steps, surface level (global_4deg_learning + jcm)")

snap_path = os.path.join(OUT_DIR, "temp_snapshot.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"saved {snap_path}")
