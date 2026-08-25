"""Report 11: bracket the gradient-trustworthy window length starting from
the post-spin-up (already turbulent) atmosphere state used in report-10,
instead of report-3's cold day-0 start.

Report 10 found n=20 from a 50-day-spun-up state is already unreliable
(87.5% FD rel error, vs. 7.3% cold-start) -- the report-3 n<=20 result is
specific to starting from a quiescent atmosphere, not a general property
of "any 20-day window." This finds where the trustworthy threshold
actually sits once the atmosphere is already turbulent: same FD-vs-AD
spot-check methodology as report-3's bracket (d(loss)/d(c_k), single
value_and_grad + central FD at eps=1e-2), swept over small n instead of
report-3's [12, 15, 20].

Same 50-day spin-up as report-10 (forward-only, no grad), same
true c_k=0.1 test point.
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
N_STEPS_LIST = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
FD_EPS = 1e-4  # rerun at smaller, properly-scaled eps per report-13's eps-sweep finding
DTYPE = DTypePolicy(enable_x64=True)

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
    config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)),
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
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=exchanges,
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

print(f"\n{'n':>6s}  {'AD grad':>14s}  {'FD grad':>14s}  {'rel err':>10s}  {'wall_s':>8s}")
for n_steps in N_STEPS_LIST:
    t0 = time.time()
    cpl = make_coupler(calib_start, n_steps)
    cpl.initial_state()  # eager prep before jit trace, per report-6/7 lesson

    def loss_fn(c_k, cpl=cpl):
        component_state = spun_up_state._component_state("OCN")
        payload = set_veros_variable(component_state.payload, "c_k", c_k)
        state = spun_up_state._with_component_state("OCN", component_state.with_payload(payload))
        result = cpl.run(state, output=None)
        return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)

    ad_value, ad_grad = jax.value_and_grad(loss_fn)(jnp.asarray(TRUE_C_K))
    loss_plus = float(loss_fn(jnp.asarray(TRUE_C_K + FD_EPS)))
    loss_minus = float(loss_fn(jnp.asarray(TRUE_C_K - FD_EPS)))
    fd_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
    rel_err = abs(float(ad_grad) - fd_grad) / abs(fd_grad)

    dt = time.time() - t0
    print(f"{n_steps:6d}  {float(ad_grad):14.4e}  {fd_grad:14.4e}  {rel_err:10.1%}  {dt:8.1f}", flush=True)
