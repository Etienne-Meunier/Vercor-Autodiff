"""Windowed calibration, take 2: "teacher forcing" instead of carrying the
model's own (imperfect) forward state between windows.

debug_script/18 (autoregressive carry) worked cleanly for the first few
windows (loss 60.9->40.3->28.2->22.6, c_k converging nicely) but then
diverged badly (loss up to 1.5e3 by window 14, c_k overshooting to 0.152
vs true 0.1) -- NOT from within-window gradient instability (each window
is only W=10 steps, well inside the trustworthy horizon), but from
cross-window trajectory drift: the carried-forward state evolves under the
optimizer's still-wrong params, so by a few windows in it's on a
chaotically DIFFERENT trajectory than the fixed pre-computed target,
making the loss comparison increasingly meaningless (comparing two
decorrelated points on the same chaotic attractor, not "how wrong are the
params").

Fix: reset each window's starting state to the TRUE trajectory's state at
that window's start (same target trajectory precomputed once), not the
model's own evolving state. This is "teacher forcing" (standard in
sequence-model training) / "cycling" in operational data assimilation.
Every window becomes an independent, clean W-step calibration problem
starting from ground truth -- no cross-window error compounding possible.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime, timedelta

import jax.numpy as jnp
import optax

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, ATMOSPHERE_TO_VEROS_FORCING_FIELDS, JCM_LAND_TO_ATMOSPHERE_FIELDS, OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

W = 10
N_WINDOWS = 15   # same scale as debug_script/18 for a direct comparison
LEARNING_RATE = 2e-2

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4

ocn = make_veros_gcm(config=VerosConfig(setup="global_4deg_learning", uses_atmosphere_forcing=True, restore_to_climatology=True, jitted=True, execution="jax"))
jcm_setup = make_jcm_land_atmosphere(ocn.grid, config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)))
lnd = jcm_setup.land
atm = jcm_setup.atmosphere
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def make_window_coupler(start_date, n_steps):
    clock = Clock(start=start_date, dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    return Coupler(clock=clock, components=[ocn, lnd, atm], exchanges=exchanges, run_order=["OCN", "LND", "ATM"],
                    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)))


window_starts = [datetime(2000, 1, 3) + timedelta(days=i * W) for i in range(N_WINDOWS)]
window_couplers = [make_window_coupler(s, W) for s in window_starts]
for cpl in window_couplers:
    cpl.initial_state()

base_state = window_couplers[0].initial_state()


def rollout_window(cpl, state, c_k, c_eps):
    component_state = state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state2 = state._with_component_state("OCN", component_state.with_payload(payload))
    return cpl.run(state2, output=None)


# --- generate chained TRUE trajectory (used both as target AND as each
# window's teacher-forced starting state) ---
print(f"Generating {N_WINDOWS}-window true trajectory (W={W} steps/window, true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...")
true_states = [base_state]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()
    next_state = jax.jit(lambda s, cpl=cpl: rollout_window(cpl, s, jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))(true_states[-1])
    true_states.append(jax.lax.stop_gradient(next_state))
    print(f"  window {i} target done ({time.time()-t0:.1f}s)")
target_temps = [s._component_state("OCN").payload.variables.temp for s in true_states[1:]]

# --- teacher-forced windowed optimization: each window starts from
# true_states[i] (ground truth), not from the model's own prior output ---
params = {"c_k": jnp.asarray(INIT_C_K), "c_eps": jnp.asarray(INIT_C_EPS)}
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)

print(f"\n{'window':>6s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}")
print(f"{'init':>6s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")

history = [(INIT_C_K, INIT_C_EPS, None)]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()
    start_state = true_states[i]  # teacher forcing: always start from ground truth

    def loss_fn(p, cpl=cpl, start_state=start_state, target=target_temps[i]):
        result = rollout_window(cpl, start_state, p["c_k"], p["c_eps"])
        return jnp.sum((result._component_state("OCN").payload.variables.temp - target) ** 2)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    history.append((ck_val, ceps_val, float(loss_val)))
    print(f"{i:6d}  {float(loss_val):14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  ({time.time()-t0:.1f}s)")

print("\nrecovered c_k:  ", history[-1][0], " (true:", TRUE_C_K, ")")
print("recovered c_eps:", history[-1][1], " (true:", TRUE_C_EPS, ")")
print(f"\ntotal simulated span covered by targets: {N_WINDOWS * W} days, each gradient call spanned only {W} steps")
