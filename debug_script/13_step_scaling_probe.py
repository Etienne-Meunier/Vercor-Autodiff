"""Escalating rollout-length probe: for each N_STEPS in the sweep, build a
fresh coupled OCN+LND+ATM(jcm) model at that length, compute
value_and_grad(c_k) once, compare to a 2-point central finite difference,
and report elapsed time + whether the gradient is finite/sane.

Cheap relative to a full fit (one grad call, not an optimization loop) --
the point is to find where gradient correctness breaks down (NaN, or wild
disagreement with FD) as the rollout gets longer, per
Veros-Autodiff's report-longrollouts-1.md finding that this happens well
before compute/memory limits do for chaotic coupled dynamics.

Stops the sweep early (breaks) on the first NaN or >50% relative FD error,
so we don't burn time on lengths already known to be broken.
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

STEP_SWEEP = [25, 50, 100, 250, 500, 1000]
C_K_TEST = 0.1
FD_EPS = 1e-3


def build_coupler(n_steps):
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
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere
    clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    exchanges = (
        Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
        Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
        Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
        Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
    )
    cpl = Coupler(
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
    )
    return cpl


def loss_fn_for(cpl, initial_state):
    def loss_fn(c_k):
        component_state = initial_state._component_state("OCN")
        payload = set_veros_variable(component_state.payload, "c_k", c_k)
        state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
        result = cpl.run(state, output=None)
        return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)

    return loss_fn


results = []
for n_steps in STEP_SWEEP:
    print(f"\n{'='*60}\nN_STEPS = {n_steps}\n{'='*60}")
    t0 = time.time()
    cpl = build_coupler(n_steps)
    initial_state = cpl.initial_state()
    loss_fn = loss_fn_for(cpl, initial_state)

    value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(C_K_TEST))
    value.block_until_ready()

    value_plus = loss_fn(jnp.asarray(C_K_TEST + FD_EPS))
    value_minus = loss_fn(jnp.asarray(C_K_TEST - FD_EPS))
    fd_grad = (value_plus - value_minus) / (2 * FD_EPS)

    elapsed = time.time() - t0
    is_nan = bool(jnp.isnan(grad)) or bool(jnp.isnan(fd_grad))
    rel_err = float("nan") if is_nan else abs(float(grad) - float(fd_grad)) / (abs(float(fd_grad)) + 1e-30)

    print(f"\n--- N_STEPS={n_steps} result ---")
    print(f"elapsed:  {elapsed:.1f}s")
    print(f"value:    {float(value)}")
    print(f"grad:     {float(grad)}")
    print(f"fd_grad:  {float(fd_grad)}")
    print(f"rel_err:  {rel_err}")
    print(f"nan:      {is_nan}")

    results.append((n_steps, elapsed, float(value), float(grad), float(fd_grad), rel_err, is_nan))

    if is_nan or (not is_nan and rel_err > 0.5):
        print(f"\n*** STOPPING SWEEP: N_STEPS={n_steps} {'produced NaN' if is_nan else f'rel_err={rel_err:.2%} > 50%'} ***")
        break

    del cpl, initial_state, loss_fn
    jax.clear_caches()

print(f"\n\n{'='*60}\nSWEEP SUMMARY\n{'='*60}")
print(f"{'N_STEPS':>8s}  {'elapsed(s)':>10s}  {'grad':>14s}  {'fd_grad':>14s}  {'rel_err':>10s}  {'nan':>5s}")
for n_steps, elapsed, value, grad, fd_grad, rel_err, is_nan in results:
    print(f"{n_steps:8d}  {elapsed:10.1f}  {grad:14.4e}  {fd_grad:14.4e}  {rel_err:10.4f}  {str(is_nan):>5s}")
