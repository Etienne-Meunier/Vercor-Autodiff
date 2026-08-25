"""FD eps sweep at N_STEPS=25, to check whether debug_script/13's single-eps
(1e-3) FD check (93% rel err vs AD) reflects a real AD gradient problem, or
just a badly-calibrated FD eps -- following Veros-Autodiff's own lesson
(report-2/section3b's eps-selection note) that a single blind eps choice
can be unreliable, especially as loss scale/curvature changes with rollout
length. If AD agrees with FD at some eps but not others, the FD estimate
was the problem, not the gradient.
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

N_STEPS = 25
C_K_TEST = 0.1
EPS_VALUES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

ocn = make_veros_gcm(config=VerosConfig(setup="global_4deg_learning", uses_atmosphere_forcing=True, restore_to_climatology=True, jitted=True, execution="jax"))
jcm_setup = make_jcm_land_atmosphere(ocn.grid, config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)))
lnd = jcm_setup.land
atm = jcm_setup.atmosphere
clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=N_STEPS, calendar="noleap")
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)
cpl = Coupler(clock=clock, components=[ocn, lnd, atm], exchanges=exchanges, run_order=["OCN", "LND", "ATM"],
              runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)))
initial_state = cpl.initial_state()


def loss_fn(c_k):
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return jnp.sum(result._component_state("OCN").payload.variables.temp ** 2)


print("computing AD grad ...")
t0 = time.time()
value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(C_K_TEST))
print(f"AD grad: {float(grad)}  value: {float(value)}  ({time.time()-t0:.1f}s)")

print(f"\n{'eps':>10s}  {'fd_grad':>16s}  {'rel_err_vs_AD':>14s}")
for eps in EPS_VALUES:
    t0 = time.time()
    value_plus = loss_fn(jnp.asarray(C_K_TEST + eps))
    value_minus = loss_fn(jnp.asarray(C_K_TEST - eps))
    fd_grad = (value_plus - value_minus) / (2 * eps)
    rel_err = abs(float(grad) - float(fd_grad)) / (abs(float(fd_grad)) + 1e-30)
    print(f"{eps:10.1e}  {float(fd_grad):16.6e}  {rel_err:14.4f}  ({time.time()-t0:.1f}s)")
