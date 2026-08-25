"""Bracket where gradient/FD agreement breaks down between the confirmed-
good N_STEPS=10 (report-2, clean fit) and confirmed-bad N_STEPS=25
(debug_script/14: FD unstable and sign-flipping across eps -- chaotic
trajectory divergence, not a bad eps choice). Tests N_STEPS in [12, 15, 20]
with two eps values each (1e-2, 1e-3) -- if FD agrees across both eps at a
given N_STEPS, gradients are still trustworthy there; if FD disagrees
between the two eps (or with AD), chaos has already set in.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

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

C_K_TEST = 0.1
EPS_VALUES = [1e-2, 1e-3]
N_STEPS_LIST = [12, 15, 20]


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
for n_steps in N_STEPS_LIST:
    print(f"\n{'='*60}\nN_STEPS = {n_steps}\n{'='*60}")
    cpl = build_coupler(n_steps)
    initial_state = cpl.initial_state()

    def loss_fn(c_k, cpl=cpl, initial_state=initial_state):
        component_state = initial_state._component_state("OCN")
        payload = set_veros_variable(component_state.payload, "c_k", c_k)
        state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
        result = cpl.run(state, output=None)
        return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)

    t0 = time.time()
    value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(C_K_TEST))
    print(f"AD grad: {float(grad):.4e}  value: {float(value):.4e}  ({time.time()-t0:.1f}s)")

    fd_grads = []
    for eps in EPS_VALUES:
        t0 = time.time()
        value_plus = loss_fn(jnp.asarray(C_K_TEST + eps))
        value_minus = loss_fn(jnp.asarray(C_K_TEST - eps))
        fd_grad = float((value_plus - value_minus) / (2 * eps))
        fd_grads.append(fd_grad)
        rel_err = abs(float(grad) - fd_grad) / (abs(fd_grad) + 1e-30)
        print(f"  eps={eps:.0e}  fd_grad={fd_grad:14.4e}  rel_err_vs_AD={rel_err:8.4f}  ({time.time()-t0:.1f}s)")

    fd_disagree = abs(fd_grads[0] - fd_grads[1]) / (abs(fd_grads[0]) + abs(fd_grads[1]) + 1e-30)
    same_sign = (fd_grads[0] > 0) == (fd_grads[1] > 0) == (float(grad) > 0)
    print(f"  FD-vs-FD disagreement: {fd_disagree:.4f}  all same sign: {same_sign}")

    results.append((n_steps, float(grad), fd_grads[0], fd_grads[1], fd_disagree, same_sign))

    del cpl, initial_state
    jax.clear_caches()

print(f"\n\n{'='*60}\nBRACKET SUMMARY\n{'='*60}")
print(f"{'N_STEPS':>8s}  {'AD grad':>14s}  {'FD(1e-2)':>14s}  {'FD(1e-3)':>14s}  {'FD disagree':>12s}  {'sane':>5s}")
for n_steps, grad, fd1, fd2, disagree, same_sign in results:
    print(f"{n_steps:8d}  {grad:14.4e}  {fd1:14.4e}  {fd2:14.4e}  {disagree:12.4f}  {str(same_sign):>5s}")
