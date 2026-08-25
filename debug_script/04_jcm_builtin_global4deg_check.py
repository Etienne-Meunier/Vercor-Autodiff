"""Check whether the OCN->ATM SST NaN seen with setup='global_4deg_learning'
also happens with the built-in 'global_4deg' setup (as in the reference
run_jcm_with_veros.py), short rollout, no spinup, x64 on (required for jcm).
Isolates whether the NaN is specific to the _learning setup variant or a
general property of the JCM<->Veros grid pairing.
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
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veros_gcm,
)
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg",
        uses_atmosphere_forcing=True,
        restore_to_climatology=False,
        jitted=True,
        execution="jax",
    ),
)

jcm_setup = make_jcm_land_atmosphere(
    ocn.grid,
    config=JCMLandAtmosphereConfig(
        atmosphere=JAXGCMConfig(
            spinup=Spinup(enabled=False),
            jitted=True,
        ),
    ),
)
lnd = jcm_setup.land
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
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)
components = [ocn, lnd, atm]
cpl = Coupler(
    clock=clock,
    components=components,
    exchanges=exchanges,
    run_order=["OCN", "LND", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)

cpl.run(output=None)
