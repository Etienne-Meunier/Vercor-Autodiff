"""Check whether OCN's own 'sea_surface_temperature' output field (as fed
into the ATM<->OCN exchange) is NaN, and where -- before involving JCM's
grid/regridder at all. Runs OCN alone (self-forced, matching debug_script/
01) for a couple of steps and inspects the exchange-facing field directly.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RuntimeOptions
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.topology import SurfaceMaskPolicy

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
    steps=2,
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
component_state = initial_state._component_state("OCN")

sst = component_state.fields.get("sea_surface_temperature")
print("initial sea_surface_temperature: type", type(sst))
if sst is not None:
    arr = jnp.asarray(sst)
    print("shape:", arr.shape)
    print("nan fraction:", float(jnp.mean(jnp.isnan(arr))))
    print("min/max (ignoring nan):", float(jnp.nanmin(arr)), float(jnp.nanmax(arr)))

result = cpl.run(initial_state, output=None)
result_component_state = result._component_state("OCN")
sst2 = result_component_state.fields.get("sea_surface_temperature")
arr2 = jnp.asarray(sst2)
print("\nafter 2 steps sea_surface_temperature:")
print("nan fraction:", float(jnp.mean(jnp.isnan(arr2))))
print("min/max (ignoring nan):", float(jnp.nanmin(arr2)), float(jnp.nanmax(arr2)))
