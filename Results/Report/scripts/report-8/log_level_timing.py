"""Report 8 addendum: does disabling per-step INFO logging (which fires an
ordered jax.debug.callback host round-trip every scanned step/component,
see vercor/_runtime/progress.py and vercor/_logging/callback.py) speed up
the coupled rollout's forward+backward pass?

Builds two Coupler instances over the same components (log_level="INFO"
default vs. "WARNING"), times N_REPEATS value_and_grad evaluations on
each (first call includes XLA compile, so timed and reported separately
from the steady-state repeats).
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

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

N_STEPS = 20
N_REPEATS = 3  # steady-state timed calls after the first (compile) call
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

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


def time_variant(log_level: str) -> dict:
    cpl = Coupler(
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
        log_level=log_level,
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

    target_temp = jax.jit(final_ocn_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
    target_temp = jax.lax.stop_gradient(target_temp)

    def loss_fn(params):
        temp = final_ocn_temp(params["c_k"], params["c_eps"])
        return jnp.sum((temp - target_temp) ** 2)

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    params = {"c_k": jnp.asarray(0.05), "c_eps": jnp.asarray(0.4)}

    print(f"\n[{log_level}] first call (includes XLA compile) ...", flush=True)
    t0 = time.time()
    loss_value, grads = value_and_grad_fn(params)
    jax.block_until_ready((loss_value, grads))
    first_call_s = time.time() - t0
    print(f"[{log_level}] first call: {first_call_s:.1f}s  loss={float(loss_value):.4e}", flush=True)

    repeat_times = []
    for i in range(N_REPEATS):
        t0 = time.time()
        loss_value, grads = value_and_grad_fn(params)
        jax.block_until_ready((loss_value, grads))
        dt = time.time() - t0
        repeat_times.append(dt)
        print(f"[{log_level}] repeat {i}: {dt:.1f}s  loss={float(loss_value):.4e}", flush=True)

    return {"first_call_s": first_call_s, "repeat_times": repeat_times}


results = {}
for log_level in ("INFO", "WARNING"):
    results[log_level] = time_variant(log_level)

print("\n=== summary ===")
for log_level, r in results.items():
    mean_repeat = sum(r["repeat_times"]) / len(r["repeat_times"])
    print(f"{log_level}: first_call={r['first_call_s']:.1f}s  mean_repeat={mean_repeat:.1f}s  "
          f"repeats={[f'{t:.1f}' for t in r['repeat_times']]}")

info_mean = sum(results["INFO"]["repeat_times"]) / len(results["INFO"]["repeat_times"])
warn_mean = sum(results["WARNING"]["repeat_times"]) / len(results["WARNING"]["repeat_times"])
print(f"\nspeedup (INFO mean / WARNING mean): {info_mean / warn_mean:.2f}x")
