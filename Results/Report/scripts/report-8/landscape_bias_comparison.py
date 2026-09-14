"""Report 8 addendum: before/after temperature-bias comparison across the
three landscape_multistart.py runs -- init (pre-optimization) vs
best-iteration (c_k, c_eps), against the same N_STEPS=20 target used
there. Forward-only rollouts (7 total: target + 3 init + 3 best states),
no grad -- cheap compared to the optimization runs themselves.

The "init" params for each run ARE that run's start point in
landscape_multistart.py's STARTS list, so no extra bookkeeping needed
there. Reads best (c_k, c_eps) per start from
Results/Report/figures/report-8/start_{0,1,2}_trajectory.npy (written by
landscape_multistart.py).
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
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
Z_LEVEL = -1  # surface level

STARTS = [(0.05, 0.40), (0.15, 0.85), (0.03, 0.55)]

# --- load each start's best-iteration (c_k, c_eps) from its checkpoint ---
best_params = []
for idx, (sck, sce) in enumerate(STARTS):
    traj = np.load(os.path.join(OUT_DIR, f"start_{idx}_trajectory.npy"))
    iters = traj[1:]  # drop the init row (loss=nan)
    best_row = iters[np.argmin(iters[:, 2])]
    best_ck, best_ceps, best_loss = float(best_row[0]), float(best_row[1]), float(best_row[2])
    print(f"start ({sck}, {sce}): best c_k={best_ck:.4f}  c_eps={best_ceps:.4f}  loss={best_loss:.4e}")
    best_params.append((sck, sce, best_ck, best_ceps, best_loss))

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


final_ocn_payload_jit = jax.jit(final_ocn_payload)


def field_at_level(payload, z):
    tau = payload.variables.tau
    temp = np.asarray(payload.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(payload.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


TARGET_PATH = os.path.join(OUT_DIR, "bias_comparison_target_field.npy")
XT_PATH = os.path.join(OUT_DIR, "bias_comparison_xt.npy")
YT_PATH = os.path.join(OUT_DIR, "bias_comparison_yt.npy")
BEST_BIAS_PATHS = [os.path.join(OUT_DIR, f"bias_comparison_bias_{i}.npy") for i in range(3)]
INIT_BIAS_PATHS = [os.path.join(OUT_DIR, f"bias_comparison_init_bias_{i}.npy") for i in range(3)]

if os.path.exists(TARGET_PATH) and all(os.path.exists(p) for p in BEST_BIAS_PATHS + INIT_BIAS_PATHS):
    print("\nreusing cached rollout results (delete bias_comparison_*.npy in figures/report-8/ to force a recompute)")
    target_field = np.load(TARGET_PATH)
    xt = np.load(XT_PATH)
    yt = np.load(YT_PATH)
    best_biases = [np.load(p) for p in BEST_BIAS_PATHS]
    init_biases = [np.load(p) for p in INIT_BIAS_PATHS]
else:
    print(f"\nrolling out target (c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...")
    target_payload = final_ocn_payload_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
    xt = np.asarray(ocn.grid.longitude)
    yt = np.asarray(ocn.grid.latitude)
    target_field = field_at_level(target_payload, Z_LEVEL)
    np.save(TARGET_PATH, target_field)
    np.save(XT_PATH, xt)
    np.save(YT_PATH, yt)

    best_biases = []
    for i, (sck, sce, best_ck, best_ceps, best_loss) in enumerate(best_params):
        print(f"rolling out start ({sck}, {sce}) best params (c_k={best_ck:.4f}, c_eps={best_ceps:.4f}) ...")
        payload = final_ocn_payload_jit(jnp.asarray(best_ck), jnp.asarray(best_ceps))
        field = field_at_level(payload, Z_LEVEL)
        bias = field - target_field
        best_biases.append(bias)
        np.save(BEST_BIAS_PATHS[i], bias)

    init_biases = []
    for i, (sck, sce) in enumerate(STARTS):
        print(f"rolling out start ({sck}, {sce}) init (pre-optimization) params ...")
        payload = final_ocn_payload_jit(jnp.asarray(sck), jnp.asarray(sce))
        field = field_at_level(payload, Z_LEVEL)
        bias = field - target_field
        init_biases.append(bias)
        np.save(INIT_BIAS_PATHS[i], bias)

all_biases = best_biases + init_biases
all_abs = np.abs(np.concatenate([b[~np.isnan(b)] for b in all_biases]))
# robust color range: a handful of single-cell noise spikes (e.g. one pixel at
# 0.52 vs. a 99.5th percentile of ~0.08) otherwise wash out the real signal
# under a flat pale color across every panel. Clipping to the 99.5th
# percentile saturates those rare spikes instead of letting them set the scale.
bias_max = np.percentile(all_abs, 99.5)

fig, axs = plt.subplots(2, 4, figsize=(21, 8.5), sharey=True, sharex=True)
im0 = axs[0, 0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[1, 0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[0, 0].set_title(f"target (surface, c_k={TRUE_C_K}, c_eps={TRUE_C_EPS})")
axs[1, 0].set_title("target (repeated for row alignment)")
fig.colorbar(im0, ax=axs[0, 0], label="temp", shrink=0.8)

for ax, bias, (sck, sce) in zip(axs[0, 1:], init_biases, STARTS):
    im = ax.pcolormesh(xt, yt, bias.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
    ax.set_title(f"BEFORE: init ({sck}, {sce}) - target")

for ax, bias, (sck, sce, best_ck, best_ceps, best_loss) in zip(axs[1, 1:], best_biases, best_params):
    im = ax.pcolormesh(xt, yt, bias.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
    ax.set_title(f"AFTER: best (c_k={best_ck:.4f}, c_eps={best_ceps:.4f})\nloss={best_loss:.3f} - target")

fig.subplots_adjust(top=0.88, right=0.90, wspace=0.12, hspace=0.35)
cax = fig.add_axes([0.92, 0.15, 0.015, 0.6])
fig.colorbar(im, cax=cax, label="bias")

for ax in axs[1]:
    ax.set_xlabel("xt")
for ax in axs[:, 0]:
    ax.set_ylabel("yt")
fig.suptitle(f"Report 8: before/after temperature bias, 3 multistart runs (N_STEPS={N_STEPS})")

out_path = os.path.join(OUT_DIR, "landscape_bias_comparison.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nsaved {out_path}")
