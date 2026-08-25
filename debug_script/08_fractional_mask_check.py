"""Check whether SurfaceMaskPolicy's ocean fractional mask on the JCM/ATM
grid (computed via a *conservative* remapper, separate from the per-field
bilinear regridder) is NaN. exchange_dispatch.py multiplies every scalar
exchanged field by this mask -- if it's NaN, every field gets poisoned,
which would explain the 100% NaN sea_surface_temperature seen in the full
coupled run despite the bilinear regridder working fine standalone.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.grid_masks import create_lnd_mask_from_ocn, compute_ocn_lnd_masks_on_atm_grid
from vercor._regridders.conservative import ConservativeRectilinearRegridder

ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg_learning",
        uses_atmosphere_forcing=False,
        restore_to_climatology=False,
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

print("computing lnd mask from ocn via create_lnd_mask_from_ocn ...")
lnd_bmask, lnd_fmask = create_lnd_mask_from_ocn(atm.grid.latitude, atm.grid.longitude, ocn.grid)
print("lnd_bmask nan fraction:", float(jnp.mean(jnp.isnan(lnd_bmask))))
print("lnd_fmask nan fraction:", float(jnp.mean(jnp.isnan(lnd_fmask))))
print("lnd_fmask min/max:", float(jnp.nanmin(lnd_fmask)), float(jnp.nanmax(lnd_fmask)))

regridder = ConservativeRectilinearRegridder(ocn.grid, atm.grid)
ocn_fmask, lnd_fmask2, lnd_bmask2 = compute_ocn_lnd_masks_on_atm_grid(
    ocn.grid.binary_mask, regridder
)
print("\nocn_fmask_on_atm_grid nan fraction:", float(jnp.mean(jnp.isnan(ocn_fmask))))
print("ocn_fmask_on_atm_grid min/max:", float(jnp.nanmin(ocn_fmask)), float(jnp.nanmax(ocn_fmask)))
