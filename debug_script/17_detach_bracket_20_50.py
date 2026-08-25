"""Bracket the detached-ATM-gradient breakdown point between the confirmed-
OK n=20 (28.6% rel err) and confirmed-broken n=50 (104% rel err, sign flip)
from debug_script/16. Tests n=[30, 40] to find where it actually crosses.
Same compute_fluxes monkeypatch (stop_gradient on ATM->OCN forcing).
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
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, ATMOSPHERE_TO_VEROS_FORCING_FIELDS, JCM_LAND_TO_ATMOSPHERE_FIELDS, OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

import vercor.setups._external.veros_fluxes as _veros_fluxes

_original_compute_fluxes = _veros_fluxes.compute_fluxes


def _detached_compute_fluxes(veros_state, runtime_fields, constants, dtype):
    detached_fields = {k: jax.lax.stop_gradient(v) for k, v in runtime_fields.items()}
    return _original_compute_fluxes(veros_state, detached_fields, constants, dtype)


_veros_fluxes.compute_fluxes = _detached_compute_fluxes
print("Monkeypatched compute_fluxes: ATM->OCN forcing fields are stop_gradient'd.")

C_K_TEST = 0.1
FD_EPS = 1e-2
STEP_SWEEP = [30, 40]


def build_coupler(n_steps):
    ocn = make_veros_gcm(config=VerosConfig(setup="global_4deg_learning", uses_atmosphere_forcing=True, restore_to_climatology=True, jitted=True, execution="jax"))
    jcm_setup = make_jcm_land_atmosphere(ocn.grid, config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)))
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere
    clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    exchanges = (
        Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
        Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
        Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
        Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
    )
    cpl = Coupler(clock=clock, components=[ocn, lnd, atm], exchanges=exchanges, run_order=["OCN", "LND", "ATM"],
                  runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)))
    return cpl


results = []
for n_steps in STEP_SWEEP:
    print(f"\n{'='*60}\nN_STEPS = {n_steps} (ATM forcing detached)\n{'='*60}")
    t0 = time.time()
    cpl = build_coupler(n_steps)
    initial_state = cpl.initial_state()

    def loss_fn(c_k, cpl=cpl, initial_state=initial_state):
        component_state = initial_state._component_state("OCN")
        payload = set_veros_variable(component_state.payload, "c_k", c_k)
        state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
        result = cpl.run(state, output=None)
        return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)

    value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(C_K_TEST))
    value_plus = loss_fn(jnp.asarray(C_K_TEST + FD_EPS))
    value_minus = loss_fn(jnp.asarray(C_K_TEST - FD_EPS))
    fd_grad = (value_plus - value_minus) / (2 * FD_EPS)
    elapsed = time.time() - t0

    is_nan = bool(jnp.isnan(grad)) or bool(jnp.isnan(fd_grad))
    rel_err = float("nan") if is_nan else abs(float(grad) - float(fd_grad)) / (abs(float(fd_grad)) + 1e-30)

    print(f"\n--- N_STEPS={n_steps} (detached) result ---")
    print(f"elapsed:  {elapsed:.1f}s  value: {float(value)}  grad: {float(grad)}  fd_grad: {float(fd_grad)}  rel_err: {rel_err}  nan: {is_nan}")

    results.append((n_steps, elapsed, float(grad), float(fd_grad), rel_err, is_nan))

    del cpl, initial_state
    jax.clear_caches()

print(f"\n\n{'='*60}\nBRACKET SUMMARY (detached, n=20/30/40/50)\n{'='*60}")
print(f"{'N_STEPS':>8s}  {'grad':>14s}  {'fd_grad':>14s}  {'rel_err':>10s}")
print(f"{20:8d}  {-3643.2652:14.4e}  {-5102.3391:14.4e}  {0.2860:10.4f}   (from debug_script/16)")
for n_steps, elapsed, grad, fd_grad, rel_err, is_nan in results:
    print(f"{n_steps:8d}  {grad:14.4e}  {fd_grad:14.4e}  {rel_err:10.4f}")
print(f"{50:8d}  {-2445.3:14.4e}  {60504.0:14.4e}  {1.0404:10.4f}   (from debug_script/16)")
