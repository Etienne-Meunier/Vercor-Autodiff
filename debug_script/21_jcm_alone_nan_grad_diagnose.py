"""Diagnose report-12's AD-grad-is-NaN finding: JCM+slab (no full Veros
ocean), n=1, perturbing metadata.temperature_variation (the real spectral
prognostic field, per debug_script/20's earlier fix). FD is finite and
smooth; AD is NaN at every n tested. Enables jax_debug_nans to get a
traceback pointing at the first NaN-producing op in the backward pass.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

from datetime import datetime

import jax.numpy as jnp
import dataclasses

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.setups import JAXGCMConfig, load_jcm_inputs, make_jax_gcm, make_slab_land, make_slab_ocean
from vercor.dtypes import as_jax_real_array, DTypePolicy
from vercor.regridding import bilinear, conservative
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.topology import SurfaceMaskPolicy

inputs = load_jcm_inputs()
coords, terrain, forcing = inputs.coords, inputs.terrain, inputs.forcing
atm = make_jax_gcm(coords, terrain, config=JAXGCMConfig(forcing_data=forcing, jitted=False))  # unjitted: cleaner nan traceback

ocn_binary_mask = jnp.where(as_jax_real_array(terrain.fmask) < 1, 1, 0).T
lnd_binary_mask = 1 - ocn_binary_mask
hgrid = coords.horizontal
lnd_grid = RectilinearGrid.from_coordinates(
    "LND", longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
    latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)), binary_mask=lnd_binary_mask,
)
ocn_grid = RectilinearGrid.from_coordinates(
    "OCN", longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
    latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)), binary_mask=ocn_binary_mask,
)
ocn = make_slab_ocean(ocn_grid)
lnd = make_slab_land(lnd_grid)

exchanges = (
    Exchange(source="ATM", target="OCN", fields=JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=conservative),
)
clock = Clock(start=datetime(2001, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=1, calendar="noleap")
cpl = Coupler(
    clock=clock, components=[atm, ocn, lnd], exchanges=exchanges,
    run_order=["OCN", "LND", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)
base_state = cpl.initial_state()


def loss_fn(delta):
    cs = base_state._component_state("ATM")
    meta = cs.payload.jcm_state.metadata
    new_meta = meta.replace(temperature_variation=meta.temperature_variation + delta)
    new_jcm_state = cs.payload.jcm_state.replace(metadata=new_meta)
    new_payload = dataclasses.replace(cs.payload, jcm_state=new_jcm_state)
    state = base_state._with_component_state("ATM", cs.with_payload(new_payload))
    result = cpl.run(state, output=None)
    result_temp = result._component_state("ATM").payload.jcm_state.prog.temperature
    return jnp.sum(result_temp ** 2)


print("Forward value at delta=0:", float(loss_fn(jnp.asarray(0.0))), flush=True)
print("Now computing grad with jax_debug_nans=True (will raise with a traceback if/where NaN first appears) ...", flush=True)
value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(0.0))
print("value:", float(value))
print("grad:", float(grad))
