"""Isolate NaN-gradient source: OCN component alone (no coupler exchanges,
no ATM/LND), self-forced (uses_atmosphere_forcing=False, as
global_4deg_learning's own docstring recommends), grad of sum(temp**2)
after N steps wrt initial surface temp.

If this is finite, the NaN in run_verosad_global4deg_grad.py comes from
the coupler/exchange path (ATM<->OCN bridge), not from OCN's own physics.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

from datetime import datetime

import jax
import jax.numpy as jnp

from vercor import Clock, Coupler, RuntimeOptions
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.topology import SurfaceMaskPolicy

N_STEPS = 5

ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg_learning",
        uses_atmosphere_forcing=False,
        restore_to_climatology=False,
        jitted=True,
        execution="jax",
    ),
)

clock = Clock(
    start=datetime(2000, 1, 1, 0, 0, 0),
    dt_seconds=86400.0,
    steps=N_STEPS,
    calendar="noleap",
)

cpl = Coupler(
    clock=clock,
    components=[ocn],
    exchanges=(),
    run_order=["OCN"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
)

initial_state = cpl.initial_state()


def final_temp_sq_sum(temp_value: jax.Array) -> jax.Array:
    component_state = initial_state._component_state("OCN")
    baseline_temp = component_state.payload.variables.temp
    new_temp = baseline_temp.at[:, :, -1, :].set(temp_value)
    new_payload = set_veros_variable(component_state.payload, "temp", new_temp)
    state = initial_state._with_component_state("OCN", component_state.with_payload(new_payload))

    result = cpl.run(state, output=None)
    result_payload = result._component_state("OCN").payload
    return jnp.sum(result_payload.variables.temp ** 2)


grad_fn = jax.value_and_grad(final_temp_sq_sum)
value, gradient = grad_fn(jnp.asarray(7.0))

print("gradient (autodiff):   ", gradient)
print("value:                 ", value)

eps = 1e-2
value_plus = final_temp_sq_sum(jnp.asarray(7.0 + eps))
value_minus = final_temp_sq_sum(jnp.asarray(7.0 - eps))
fd_gradient = (value_plus - value_minus) / (2 * eps)

rel_error = abs(gradient - fd_gradient) / abs(fd_gradient)
print("gradient (finite diff):", fd_gradient)
print("relative error:        ", rel_error)
