"""Directly test the OCN grid -> JCM/ATM grid bilinear regridder on a known
finite SST field (skip the full coupled rollout -- cheap and fast). If this
alone produces an all-NaN target field, the bug is in grid construction or
the regridder, not in Veros/JCM physics.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.regridding import bilinear

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

print("OCN grid lon range:", float(ocn.grid.longitude.min()), float(ocn.grid.longitude.max()))
print("OCN grid lat range:", float(ocn.grid.latitude.min()), float(ocn.grid.latitude.max()))
print("OCN grid shape:", ocn.grid.longitude.shape, ocn.grid.latitude.shape)
print("ATM grid lon range:", float(atm.grid.longitude.min()), float(atm.grid.longitude.max()))
print("ATM grid lat range:", float(atm.grid.latitude.min()), float(atm.grid.latitude.max()))
print("ATM grid shape:", atm.grid.longitude.shape, atm.grid.latitude.shape)

# A plausible finite SST-like field on the OCN grid: 285 K everywhere the
# ocean mask is set, NaN over land (matches what extract_veros_runtime_sst
# would realistically produce before/without the OCN->ATM exchange).
mask = jnp.asarray(ocn.grid.binary_mask)
print("OCN binary_mask shape/dtype:", mask.shape, mask.dtype, "fraction ocean:", float(jnp.mean(mask)))

sst = jnp.where(mask.astype(bool), 285.0, jnp.nan)

regridder = bilinear(ocn.grid, atm.grid)
out = regridder.regrid(sst)
print("\nregridded SST on ATM grid:")
print("shape:", out.shape)
print("nan fraction:", float(jnp.mean(jnp.isnan(out))))
finite = out[jnp.isfinite(out)]
if finite.size:
    print("min/max:", float(finite.min()), float(finite.max()))
else:
    print("ALL NaN")
