"""Report 13: lead investigation -- is report-11's post-spinup breakdown
(n=3: 193.9% rel err, sign flip) an FD-eps-scale artifact, or genuine
chaos-onset? Sweeps eps at fixed n=3 from the same 50-day-spun-up state
used in report-10/11, mirroring report-3's debug_script/14 eps-sweep at
n=25 cold-start.

If it's a genuine chaotic-decorrelation signature, the FD estimate should
not converge (and may not hold a consistent sign) as eps shrinks -- same
signature report-3 found cold-start at n=25.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))
sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime, timedelta

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

START = datetime(2001, 1, 3, 0, 0, 0)
SPINUP_DAYS = 50
N_STEPS = 3
EPS_LIST = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
TRUE_C_K = 0.1
DTYPE = DTypePolicy(enable_x64=True)

ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg_learning", uses_atmosphere_forcing=True,
        restore_to_climatology=True, jitted=True, execution="jax",
    ),
)
jcm_setup = make_jcm_land_atmosphere(
    ocn.grid, config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)),
)
lnd = jcm_setup.land
atm = jcm_setup.atmosphere
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def make_coupler(start_date, n_steps):
    clock = Clock(start=start_date, dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    return Coupler(
        clock=clock, components=[ocn, lnd, atm], exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTYPE),
    )


print(f"Spinning up {SPINUP_DAYS} days forward-only (no grad) from {START} ...", flush=True)
spinup_cpl = make_coupler(START, SPINUP_DAYS)
t0 = time.time()
spun_up_state = jax.jit(lambda s: spinup_cpl.run(s, output=None))(spinup_cpl.initial_state())
spun_up_state = jax.lax.stop_gradient(spun_up_state)
print(f"spin-up done ({time.time() - t0:.1f}s)", flush=True)

calib_start = START + timedelta(days=SPINUP_DAYS)
cpl = make_coupler(calib_start, N_STEPS)
cpl.initial_state()


def loss_fn(c_k):
    component_state = spun_up_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    state = spun_up_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)


ad_value, ad_grad = jax.value_and_grad(loss_fn)(jnp.asarray(TRUE_C_K))
print(f"\nAD grad (n={N_STEPS}, post-spinup): {float(ad_grad):.4e}\n")

print(f"{'eps':>10s}  {'FD grad':>14s}  {'rel err':>10s}  {'wall_s':>8s}")
for eps in EPS_LIST:
    t0 = time.time()
    loss_plus = float(loss_fn(jnp.asarray(TRUE_C_K + eps)))
    loss_minus = float(loss_fn(jnp.asarray(TRUE_C_K - eps)))
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    rel_err = abs(float(ad_grad) - fd_grad) / abs(fd_grad)
    dt = time.time() - t0
    print(f"{eps:10.1e}  {fd_grad:14.4e}  {rel_err:10.1%}  {dt:8.1f}", flush=True)
