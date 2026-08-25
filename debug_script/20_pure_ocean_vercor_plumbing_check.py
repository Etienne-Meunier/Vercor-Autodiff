"""Plumbing check: does vercor's own Coupler/checkpointing wrap the same
physics Veros-Autodiff used in report-longrollouts-1.md (GlobalFourDegreeSetup,
prescribed climatological forcing, no live atmosphere) with the same
stability, or does vercor's own machinery introduce different fragility?

Single OCN component, no ATM/LND coupling (uses_atmosphere_forcing=False,
restore_to_climatology=True -- same prescribed-forcing config the "_learning"
setup uses when not coupled to a live atmosphere), run through vercor's own
Coupler exactly as debug_script/01 and examples/run_verosad_grad.py do it,
but with global_4deg_learning instead of the tiny acc setup, matching Report
3's d(loss)/d(c_k) methodology (loss = sum(temp**2) after N_STEPS, grad wrt
c_k) so it's directly comparable to Report 3's n=12..25 numbers.

n-sweep is AD-only (cheap: one value_and_grad call per n) with FD spot
checks (properly-scaled eps=1e-2, per Report 3's lesson) at a couple of
points for sanity, not a full sweep -- goal here is just: does the AD
gradient stay smooth/bounded well past n=25 (where the coupled JCM model
broke down), consistent with Autodiff's own ~1000-step horizon under the
same forcing type?

Starts conservative (n up to 500) rather than jumping straight to n=1000,
since vercor's coupler checkpoints with a single-level jax.checkpoint
around the whole lax.scan (no outer chunking) -- Autodiff's own
report-longrollouts-1 needed a double_checkpoint (chunked
scan-of-checkpointed-scans) specifically to avoid OOM past ~n=400 on a
16GB GPU; vercor doesn't have that, so memory behavior at n=500-1000 here
is unknown and worth checking incrementally.
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

from vercor import Clock, Coupler, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable

TRUE_C_K = 0.1
N_STEPS_LIST = [20, 25, 50, 100, 250, 500]
FD_CHECK_STEPS = {20, 50, 250}
FD_EPS = 1e-2


def build_coupler(n_steps):
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=False,
            restore_to_climatology=True,
            jitted=True,
            execution="jax",
        ),
    )
    clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    cpl = Coupler(clock=clock, components=[ocn], run_order=["OCN"], runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)))
    return cpl


def loss_fn(cpl, initial_state, c_k):
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    temp = result._component_state("OCN").payload.variables.temp
    return jnp.sum(temp ** 2)


print(f"{'n':>6s}  {'AD grad':>14s}  {'FD grad':>14s}  {'rel err':>10s}  {'wall_s':>8s}")
for n_steps in N_STEPS_LIST:
    t0 = time.time()
    cpl = build_coupler(n_steps)
    initial_state = cpl.initial_state()

    value_and_grad_fn = jax.value_and_grad(lambda ck: loss_fn(cpl, initial_state, ck))
    value, grad = value_and_grad_fn(jnp.asarray(TRUE_C_K))
    grad = float(grad)

    fd_str, err_str = "--", "--"
    if n_steps in FD_CHECK_STEPS:
        loss_plus = float(loss_fn(cpl, initial_state, jnp.asarray(TRUE_C_K + FD_EPS)))
        loss_minus = float(loss_fn(cpl, initial_state, jnp.asarray(TRUE_C_K - FD_EPS)))
        fd_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
        rel_err = abs(grad - fd_grad) / abs(fd_grad)
        fd_str, err_str = f"{fd_grad:.4e}", f"{rel_err:.1%}"

    dt = time.time() - t0
    print(f"{n_steps:6d}  {grad:14.4e}  {fd_str:>14s}  {err_str:>10s}  {dt:8.1f}", flush=True)
