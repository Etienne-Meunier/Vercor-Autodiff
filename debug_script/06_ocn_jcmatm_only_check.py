"""OCN + JCM-atmosphere only (no LND), short rollout. Isolates whether LND
involvement is required to reproduce the all-NaN sea_surface_temperature
seen in the full OCN+LND+ATM jcm grad script.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
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

clock = Clock(
    start=datetime(2000, 1, 3, 0, 0, 0),
    dt_seconds=86400.0,
    steps=2,
    calendar="noleap",
)

exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
)

cpl = Coupler(
    clock=clock,
    components=[ocn, atm],
    exchanges=exchanges,
    run_order=["OCN", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)

cpl.run(output=None)
