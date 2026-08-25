"""Report 12: lead investigation for "why is our post-spinup gradient
horizon (report-11: n<=2) so much shorter than a referenced paper's JCM
calibration over 5 steps?" Tests JCM alone (JCM atmosphere + toy slab
ocean + toy slab land, no full 3D Veros ocean circulation) instead of
our full OCN(Veros)+LND+ATM(jcm) coupling used in reports 1-11.

Setup mirrors examples/run_jcm_with_slab.py (make_jax_gcm + make_slab_ocean
+ make_slab_land) -- the toy slab ocean is a mixed-layer energy-balance
model, not a chaotic circulation model, so this isolates whether JCM's own
atmospheric dynamics are as fast-chaotic as our OCN+ATM(jcm) coupling, or
whether the short horizon found in reports 3/10/11 is specific to coupling
JCM to full Veros ocean dynamics.

Gradient target: same style as examples/run_verosad_grad.py (perturb a
scalar into the initial state, differentiate the final loss w.r.t. it) --
here, a scalar offset added to JCM's initial atmospheric temperature field
(jcm_state.prog.temperature), loss = sum(temperature**2) after N_STEPS.
FD-vs-AD bracket over n = [1, 2, 3, 5, 8, 10, 15, 20], cold start (no
atmosphere spin-up, matching report-3's original cold-start recipe).
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

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.setups import JAXGCMConfig, load_jcm_inputs, make_jax_gcm, make_slab_land, make_slab_ocean
from vercor.dtypes import as_jax_real_array, DTypePolicy
from vercor.regridding import bilinear, conservative
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.topology import SurfaceMaskPolicy

START = datetime(2001, 1, 3, 0, 0, 0)
N_STEPS_LIST = [1, 2, 3, 5, 8, 10, 15, 20]
FD_EPS = 1e-2
DTYPE = DTypePolicy(enable_x64=True)

inputs = load_jcm_inputs()
coords, terrain, forcing = inputs.coords, inputs.terrain, inputs.forcing
atm = make_jax_gcm(coords, terrain, config=JAXGCMConfig(forcing_data=forcing, jitted=True))

ocn_binary_mask = jnp.where(as_jax_real_array(terrain.fmask) < 1, 1, 0).T
lnd_binary_mask = 1 - ocn_binary_mask
hgrid = coords.horizontal
lnd_grid = RectilinearGrid.from_coordinates(
    "LND", longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
    latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)), binary_mask=lnd_binary_mask,
)
ocn_grid = RectilinearGrid.from_coordinates(
    "OCN", longitude=jnp.rad2deg(as_jax_real_array(hgrid.longitudes)),
    latitude=jnp.rad2deg(as_jax_real_array(hgrid.latitudes)), binary_mask=ocn_binary_mask,
)
ocn = make_slab_ocean(ocn_grid)
lnd = make_slab_land(lnd_grid)

exchanges = (
    Exchange(source="ATM", target="OCN", fields=JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=conservative),
)


def make_coupler(n_steps):
    clock = Clock(start=START, dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    return Coupler(
        clock=clock,
        components=[atm, ocn, lnd],
        exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTYPE),
    )


def perturbed_loss(cpl, base_state, delta):
    # NOTE: the ATM step function integrates state.metadata (dinosaur's
    # spectral primitive_equations.State), not state.prog -- prog is a
    # nodal-space *diagnostic* recomputed from metadata after each step
    # (jax_gcm_state.py:131-150). Perturbing prog is therefore a no-op on
    # the actual dynamics (confirmed empirically: identical output with
    # and without a +5K prog perturbation). Perturb metadata.temperature_variation
    # instead -- the real spectral prognostic temperature field.
    cs = base_state._component_state("ATM")
    meta = cs.payload.jcm_state.metadata
    new_meta = meta.replace(temperature_variation=meta.temperature_variation + delta)
    new_jcm_state = cs.payload.jcm_state.replace(metadata=new_meta)
    import dataclasses
    new_payload = dataclasses.replace(cs.payload, jcm_state=new_jcm_state)
    state = base_state._with_component_state("ATM", cs.with_payload(new_payload))
    result = cpl.run(state, output=None)
    result_temp = result._component_state("ATM").payload.jcm_state.prog.temperature
    return jnp.sum(result_temp ** 2)


print(f"\n{'n':>6s}  {'AD grad':>16s}  {'FD grad':>16s}  {'rel err':>10s}  {'wall_s':>8s}")
for n_steps in N_STEPS_LIST:
    t0 = time.time()
    cpl = make_coupler(n_steps)
    base_state = cpl.initial_state()

    def loss_fn(delta, cpl=cpl, base_state=base_state):
        return perturbed_loss(cpl, base_state, delta)

    ad_value, ad_grad = jax.value_and_grad(loss_fn)(jnp.asarray(0.0))
    loss_plus = float(loss_fn(jnp.asarray(FD_EPS)))
    loss_minus = float(loss_fn(jnp.asarray(-FD_EPS)))
    fd_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
    rel_err = abs(float(ad_grad) - fd_grad) / abs(fd_grad) if fd_grad != 0 else float("nan")

    dt = time.time() - t0
    print(f"{n_steps:6d}  {float(ad_grad):16.4e}  {fd_grad:16.4e}  {rel_err:10.1%}  {dt:8.1f}", flush=True)
