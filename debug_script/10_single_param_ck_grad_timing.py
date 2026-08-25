"""Isolate whether the fit script's slowdown comes from setting TWO Veros
Variables (c_k, c_eps) via two chained set_veros_variable calls, vs the
already-proven-fast single-variable case (run_jcm_global4deg_grad.py sets
just "temp"). Times a single value_and_grad call for c_k alone.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import ATMOSPHERE_TO_VEROS_FORCING_FIELDS, OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

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
atm = jcm_setup.atmosphere

clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=5, calendar="noleap")
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
)
cpl = Coupler(
    clock=clock, components=[ocn, atm], exchanges=exchanges, run_order=["OCN", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)
initial_state = cpl.initial_state()


def final_temp_sq_sum_single(c_k: jax.Array) -> jax.Array:
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)


def final_temp_sq_sum_double(c_k: jax.Array, c_eps: jax.Array) -> jax.Array:
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)


print("=== single param (c_k only) ===")
t0 = time.time()
value, grad = jax.value_and_grad(final_temp_sq_sum_single)(jnp.asarray(0.1))
print(f"elapsed: {time.time()-t0:.1f}s  value={value}  grad={grad}")

print("\n=== double param (c_k, c_eps) ===")
t0 = time.time()
value, grads = jax.value_and_grad(final_temp_sq_sum_double, argnums=(0, 1))(
    jnp.asarray(0.1), jnp.asarray(0.7)
)
print(f"elapsed: {time.time()-t0:.1f}s  value={value}  grads={grads}")
