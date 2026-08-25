"""Figures for Report 1 (c_k/c_eps fit, 5-step coupled OCN+LND+ATM(jcm) rollout).

Trajectory numbers are hardcoded from the completed fit run
(examples/fit_ck_ceps_jcm_global4deg.py, N_STEPS=5, N_ITERS=15) -- no need to
recompute the fit itself. Only the three rollouts needed for the temperature
snapshot (target / initial-guess / optimized) are rerun here, each a cheap
forward-only 5-step pass (~15-70s), following report-2/section3b_ck_ceps_long_rollout.py's
snapshot pattern from Veros-Autodiff.
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

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-1")
os.makedirs(OUT_DIR, exist_ok=True)

# --- hardcoded trajectory from the completed fit run (fit_ck_ceps_jcm_global4deg.py) ---
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4
trajectory = [
    (INIT_C_K, INIT_C_EPS, None),
    (0.0700, 0.3800, 4.777100e01),
    (0.0885, 0.3679, 1.740264e01),
    (0.1033, 0.3660, 5.414537e00),
    (0.1146, 0.3711, 4.111819e00),
    (0.1228, 0.3810, 8.705785e00),
    (0.1281, 0.3932, 1.230846e01),
    (0.1311, 0.4070, 1.464067e01),
    (0.1325, 0.4219, 1.589227e01),
    (0.1323, 0.4378, 1.629843e01),
    (0.1308, 0.4542, 1.600572e01),
    (0.1282, 0.4713, 1.500525e01),
    (0.1246, 0.4888, 1.348760e01),
    (0.1197, 0.5065, 1.158327e01),
    (0.1141, 0.5243, 9.043280e00),
    (0.1083, 0.5419, 6.416322e00),
]
FINAL_C_K, FINAL_C_EPS = trajectory[-1][0], trajectory[-1][1]

Z_LEVEL = -1  # surface level

# --- coupled model (matches fit_ck_ceps_jcm_global4deg.py exactly) ---
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

clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=5, calendar="noleap")
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


def rollout_temp(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", jnp.asarray(c_k))
    payload = set_veros_variable(payload, "c_eps", jnp.asarray(c_eps))
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload


def field_at_level(payload, z):
    tau = payload.variables.tau
    temp = np.asarray(payload.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(payload.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


print("rolling out target (true c_k/c_eps) ...")
target_payload = jax.jit(rollout_temp)(TRUE_C_K, TRUE_C_EPS)

print("rolling out initial guess ...")
initial_payload = jax.jit(rollout_temp)(INIT_C_K, INIT_C_EPS)

print("rolling out optimized (recovered c_k/c_eps) ...")
optimized_payload = jax.jit(rollout_temp)(FINAL_C_K, FINAL_C_EPS)

xt = np.asarray(ocn.grid.longitude)
yt = np.asarray(ocn.grid.latitude)

target_field = field_at_level(target_payload, Z_LEVEL)
bias_initial = field_at_level(initial_payload, Z_LEVEL) - target_field
bias_optimized = field_at_level(optimized_payload, Z_LEVEL) - target_field

bias_max = np.nanmax(
    np.abs(
        np.concatenate(
            [bias_initial[~np.isnan(bias_initial)], bias_optimized[~np.isnan(bias_optimized)]]
        )
    )
)

# --- figure 1: trajectory (loss + param path) ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

iters = list(range(len(trajectory) - 1))
losses = [t[2] for t in trajectory[1:]]
axs[0].plot(iters, losses, "o-", color="tab:blue")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared temp error)")
axs[0].set_title("Loss vs iteration")
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="tab:blue", markersize=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:blue", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle("Report 1: c_k/c_eps recovery, 5-step coupled OCN+LND+ATM(jcm) rollout")
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
fig.suptitle("Report 1: temperature bias after 5 steps, surface level (global_4deg_learning + jcm)")

snap_path = os.path.join(OUT_DIR, "temp_snapshot.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"saved {snap_path}")
