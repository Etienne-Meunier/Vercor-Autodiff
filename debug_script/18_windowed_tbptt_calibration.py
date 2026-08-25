"""Windowed / truncated-BPTT calibration: chain short (W-step) windows,
each individually differentiated within the trustworthy horizon found in
Report 3/4 (n<=20), with jax.lax.stop_gradient applied to the carried-over
state between windows. This is the same idea as truncated backprop through
time for RNNs, or "weak-constraint"/windowed-adjoint methods in
operational data assimilation for chaotic systems: no single gradient call
spans more than W steps (stays numerically trustworthy), but the
*optimization* and the *simulated trajectory* can span an arbitrary total
duration (N_WINDOWS * W steps), with the optimizer's own momentum
(Adam) aggregating signal across windows the way SGD aggregates
mini-batch gradients.

Small-scale first (W=10, N_WINDOWS=3, ~30 simulated days) to verify the
mechanism works and produces sensible loss/param behavior before scaling
up the number of windows (which is "free" to add more of -- doesn't cost
more per gradient call, just more sequential calls).
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

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

W = 10           # steps per window, within the trustworthy horizon (Report 3: clean to n=20)
N_WINDOWS = 15   # total simulated span = N_WINDOWS * W = 150 days -- validated at N_WINDOWS=3 first
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

# Each Coupler lazily prepares its runtime (grid setup etc.) on first .run()
# call, and that prep does concrete bool() checks that fail under jax.jit
# trace -- force it eagerly here for every window coupler, not just the
# first one.
for cpl in window_couplers:
    cpl.initial_state()

base_state = window_couplers[0].initial_state()


def rollout_window(cpl, state, c_k, c_eps):
    component_state = state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state2 = state._with_component_state("OCN", component_state.with_payload(payload))
    return cpl.run(state2, output=None)


# --- generate chained target trajectory at true params ---
print(f"Generating {N_WINDOWS}-window true trajectory (W={W} steps/window, true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...")
true_states = [base_state]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()
    next_state = jax.jit(lambda s, cpl=cpl: rollout_window(cpl, s, jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))(true_states[-1])
    true_states.append(next_state)
    print(f"  window {i} target done ({time.time()-t0:.1f}s)")
target_temps = [jax.lax.stop_gradient(s._component_state("OCN").payload.variables.temp) for s in true_states[1:]]

# --- windowed TBPTT optimization ---
params = {"c_k": jnp.asarray(INIT_C_K), "c_eps": jnp.asarray(INIT_C_EPS)}
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)
carry_state = base_state

print(f"\n{'window':>6s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}")
print(f"{'init':>6s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")

history = [(INIT_C_K, INIT_C_EPS, None)]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()

    def loss_fn(p, cpl=cpl, carry_state=carry_state, target=target_temps[i]):
        result = rollout_window(cpl, carry_state, p["c_k"], p["c_eps"])
        loss = jnp.sum((result._component_state("OCN").payload.variables.temp - target) ** 2)
        return loss, result

    (loss_val, result), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # advance carry state using this window's forward result (computed with
    # pre-update params); detach so the NEXT window's gradient doesn't have
    # to backprop through this window too (that's the whole point: keep
    # each window's differentiated segment at W steps, not W*N_WINDOWS)
    carry_state = jax.lax.stop_gradient(result)

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    history.append((ck_val, ceps_val, float(loss_val)))
    print(f"{i:6d}  {float(loss_val):14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  ({time.time()-t0:.1f}s)")

print("\nrecovered c_k:  ", history[-1][0], " (true:", TRUE_C_K, ")")
print("recovered c_eps:", history[-1][1], " (true:", TRUE_C_EPS, ")")
print(f"\ntotal simulated span: {N_WINDOWS * W} days, each gradient call spanned only {W} steps")
