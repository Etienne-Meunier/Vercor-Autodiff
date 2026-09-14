"""Shared model setup and observables for report 21.

Report-20 fitted `c_k`/`c_eps` at N_STEPS=10 against the *instantaneous* surface
temperature of the day-10 state. Its loss curve was noisy through the first 60
iterations (3-4x excursions above the running minimum) and one unclipped run hit
a gradient outlier ~200x its neighbours at iteration 24. Report 21 asks whether
the observable is what makes the gradient noisy, by changing only the observable:

  mld       instantaneous MLD of the day-10 state              (report-16's kernel)
  temp_avg  surface temperature averaged over the day-8/9/10 states
  mld_avg   MLD averaged over the day-8/9/10 states

report-20-clip (instantaneous surface temperature) is the fourth corner of that
2x2 and is the baseline all three are read against. Everything else -- rollout
length, start point, optimizer, schedule, iteration count, clipping rule -- is
report-20's.

Averaging over the last three days needs the *intermediate* states, which
`Coupler.run` does not expose: it returns the final state only. So the rollout is
split into three chained couplers (days 0-8, 8-9, 9-10), each started at the date
the previous one ended, exactly as report-6 chains its calibration windows.
Chaining is plain function composition, so reverse-mode AD still differentiates
the whole 10-day rollout in one `value_and_grad` call. `validate_segmented.py`
checks the chained day-10 state against a monolithic 10-step run.

The MLD kernel (`get_index_mld` / `mld_from_index`) is report-16's verbatim: the
discrete level selection is kept out of the differentiated path, and the one
division is guarded by `well_defined` *before* dividing so degenerate columns
cannot put a NaN into the summed gradient.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime, timedelta

import numpy as np
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

REPO_ROOT = _REPO_ROOT

# --- report-20's recipe, unchanged ---
N_STEPS = int(os.environ.get("REPORT21_N_STEPS", 10))
N_ITERS = int(os.environ.get("REPORT21_N_ITERS", 100))
LEARNING_RATE = 2e-2          # peak lr of the cosine-decay schedule
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
# The start point is overridable so the multistart figure can drive fit.py from
# several corners on exactly this recipe rather than a reimplementation of it.
INIT_C_K = float(os.environ.get("REPORT21_INIT_C_K", 0.05))
INIT_C_EPS = float(os.environ.get("REPORT21_INIT_C_EPS", 0.4))

# Averaging window: the last AVG_WINDOW states of the rollout, i.e. days 8, 9, 10
# at N_STEPS=10.
AVG_WINDOW = int(os.environ.get("REPORT21_AVG_WINDOW", 3))

START_DATE = datetime(2000, 1, 3, 0, 0, 0)
DT_SECONDS = 86400.0

MLD_REFERENCE_DEPTH = -10.0   # m, negative down -- report-16's value
MLD_REFERENCE_OFFSET = 0.03   # kg/m^3 density offset defining the mixed layer base

# Report-20 clipped the global gradient norm at 50 after an outlier ~200x its
# neighbours walked a run off course through adam's momentum. 50 is not portable
# across observables: an MLD loss is in m^2 and a temperature loss in K^2, so
# their gradient norms differ by orders of magnitude and a fixed 50 would either
# do nothing or clip every step. Report-20's threshold sat at 50 / 40.97 = 1.2205
# times the gradient norm at its own starting point (dL/dc_k = -4.096e1,
# dL/dc_eps = 9.704e-1 at c_k = 0.05, c_eps = 0.40), so that ratio is what
# transfers. Each run here measures its own gradient at the start point and clips
# at CLIP_FACTOR times that norm, which reproduces report-20's 50 exactly for
# report-20's observable.
CLIP_FACTOR = float(os.environ.get("REPORT21_CLIP_FACTOR", 1.2205))


def _components():
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
    return ocn, jcm_setup.land, jcm_setup.atmosphere


_EXCHANGES = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def build(segmented):
    """Model plus the grid constants the observables need.

    `segmented=False` gives one N_STEPS-step coupler (report-15/16/20's setup).
    `segmented=True` gives the chain [N_STEPS-AVG_WINDOW+1, 1, ..., 1] whose
    segment ends are exactly the states the time average runs over.
    """
    ocn, lnd, atm = _components()

    def make_coupler(start, n_steps):
        clock = Clock(start=start, dt_seconds=DT_SECONDS, steps=n_steps, calendar="noleap")
        return Coupler(
            clock=clock,
            components=[ocn, lnd, atm],
            exchanges=_EXCHANGES,
            run_order=["OCN", "LND", "ATM"],
            runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
            log_level="WARNING",  # per-step INFO logging fires a host callback per step
        )

    if segmented:
        first = N_STEPS - AVG_WINDOW + 1
        if first < 1:
            raise SystemExit(f"AVG_WINDOW={AVG_WINDOW} does not fit in N_STEPS={N_STEPS}")
        lengths = [first] + [1] * (AVG_WINDOW - 1)
    else:
        lengths = [N_STEPS]

    couplers, offset = [], 0
    for n in lengths:
        couplers.append(make_coupler(START_DATE + timedelta(seconds=offset * DT_SECONDS), n))
        offset += n
    initial_state = couplers[0].initial_state()
    for c in couplers[1:]:
        c.initial_state()  # force preparation, as report-6 does

    v = initial_state._component_state("OCN").payload.variables
    zt = np.asarray(v.zt)
    maskT = np.asarray(v.maskT[2:-2, 2:-2, :]).astype(bool)
    grid = {
        "zt": zt,
        "zt_j": jnp.asarray(zt),
        "maskT": maskT,
        "maskT_j": jnp.asarray(maskT),
        "surf": maskT.shape[2] - 1,
        "longitude": np.asarray(ocn.grid.longitude),
        "latitude": np.asarray(ocn.grid.latitude),
    }
    print(f"rollout: {len(lengths)} coupler(s), steps {lengths}, "
          f"days {[0] + list(np.cumsum(lengths))}", flush=True)
    return couplers, initial_state, grid


def make_payloads_fn(couplers, initial_state):
    """(c_k, c_eps) -> the OCN payload at the end of every segment.

    `c_k`/`c_eps` are re-set on each segment's input state. The value written is
    the same traced parameter every time, so this is an identity on the values and
    changes no gradient, but it keeps the parameters authoritative even if a
    component were to reset them internally between runs.
    """

    def payloads(c_k, c_eps):
        state = initial_state
        out = []
        for cpl in couplers:
            cs = state._component_state("OCN")
            payload = set_veros_variable(cs.payload, "c_k", c_k)
            payload = set_veros_variable(payload, "c_eps", c_eps)
            state = cpl.run(state._with_component_state("OCN", cs.with_payload(payload)), output=None)
            out.append(state._component_state("OCN").payload)
        return out

    return payloads


# --- observables ---

def surface_temp_of_payload(payload, surf):
    return payload.variables.temp[2:-2, 2:-2, surf, payload.variables.tau]


def ts_diff_of_payload(payload):
    """Top-3 vertical T and S differences to the surface level, shape (nx, ny, 4).

    Channels: T[-2]-T[-1], T[-3]-T[-1], S[-2]-S[-1], S[-3]-S[-1]. On this grid the
    MLD is a function of exactly these levels (report-22's crossing sits between
    levels -1/-2 in 95% of columns and -2/-3 in almost all the rest), so this is
    the information the MLD sees, without the density combination and the 1/drho
    transform. Differences rather than levels: the TKE signal is in the vertical
    structure, while the level values are dominated by the surface-flux common mode.
    """
    v = payload.variables
    t = v.temp[2:-2, 2:-2, -3:, v.tau]   # levels [-3, -2, -1]
    s = v.salt[2:-2, 2:-2, -3:, v.tau]
    return jnp.stack([t[..., 1] - t[..., 2], t[..., 0] - t[..., 2],
                      s[..., 1] - s[..., 2], s[..., 0] - s[..., 2]], axis=-1)


def loss_weights(kind, target, valid):
    """Per-term weights of the squared error.

    1 for the single-field observables. For tsdiff_avg each of the four channels is
    weighted 1/(n_c * var_c), with n_c and var_c the cell count and spatial variance
    of the *target* field on its valid cells -- report-17's variance normalisation,
    so the loss is the sum of four normalised mean-squared errors. The weights
    depend on the target only, so every grid node and every start descends the
    identical loss.
    """
    if kind != "tsdiff_avg":
        return 1.0
    t, m = np.asarray(target), np.asarray(valid)
    return jnp.asarray([1.0 / (m[..., c].sum() * t[..., c][m[..., c]].var()) for c in range(t.shape[-1])])


def get_index_mld(prho, maskT, zt, reference_depth, reference_offset=MLD_REFERENCE_OFFSET):
    """Level indices bracketing the MLD -- discrete selection only, no gradient path.

    Report-16's kernel verbatim. ridx is recomputed from zt rather than sliced to a
    dynamic length, so this stays jit-safe; i_below/i_above come out of argmax/argmin
    gated by boolean comparisons on prho, which carry no gradient. Land is excluded
    via maskT (authoritative regardless of eq_of_state_type), not isnan(prho) --
    under nonlin2 land cells are exactly 0.0, not NaN.
    """
    level = jnp.arange(zt.shape[-1])
    ridx = jnp.max(jnp.where(zt < reference_depth, level, -1))

    prho_reference = prho[:, :, ridx] + reference_offset
    valid = maskT.astype(bool) & (level <= ridx)

    drho = prho - prho_reference[:, :, jnp.newaxis]

    below_mask = valid & (drho > 0)
    has_below = jnp.any(below_mask, axis=-1)
    i_below = jnp.argmax(jnp.where(below_mask, zt, -jnp.inf), axis=-1)
    depth_below = zt[i_below]

    above_mask = valid & (drho < 0) & (zt > depth_below[:, :, jnp.newaxis])
    has_above = jnp.any(above_mask, axis=-1)
    i_above = jnp.argmin(jnp.where(above_mask, zt, jnp.inf), axis=-1)

    well_defined = has_below & has_above
    return ridx, i_below, i_above, well_defined


def mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset=MLD_REFERENCE_OFFSET):
    """MLD from precomputed level indices -- plain gather + arithmetic, differentiable.

    Returns 0.0 at degenerate columns rather than NaN; callers mask with
    `well_defined`. A NaN there would be fine for plotting but poisons the gradient
    of any sum over the field.
    """
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below = jnp.take_along_axis(prho, i_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    prho_above = jnp.take_along_axis(prho, i_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    depth_below = zt[i_below]
    depth_above = zt[i_above]

    denom = jnp.where(well_defined, prho_above - prho_below, 1.0)
    mld = (prho_reference - prho_below) / denom * (depth_above - depth_below) + depth_below
    return jnp.where(well_defined, mld, 0.0)


def mld_of_payload(payload, maskT, zt):
    prho = payload.variables.prho[2:-2, 2:-2, :]
    ridx, i_below, i_above, well_defined = get_index_mld(prho, maskT, zt, MLD_REFERENCE_DEPTH)
    return mld_from_index(prho, zt, ridx, i_below, i_above, well_defined), well_defined


def make_observable(kind, couplers, initial_state, grid):
    """(c_k, c_eps) -> (field, valid_mask) for one of the four observables.

    `valid_mask` is all-True for the temperature observables (the ocean mask is a
    grid constant there) and the well-defined-column mask for the MLD ones. For the
    averaged MLD it is the intersection over the averaged states: a column that
    loses its MLD on any one of the three days is dropped, so the averaged field is
    a mean of three defined values or nothing.
    """
    payloads_fn = make_payloads_fn(couplers, initial_state)
    surf = grid["surf"]
    maskT, zt = grid["maskT_j"], grid["zt_j"]
    ocean = jnp.asarray(grid["maskT"][:, :, surf])
    m2, m3 = grid["maskT"][:, :, surf - 1], grid["maskT"][:, :, surf - 2]
    tsdiff_valid = jnp.asarray(np.stack([m2, m3, m2, m3], axis=-1))  # static: no mask flips

    def observable(c_k, c_eps):
        payloads = payloads_fn(c_k, c_eps)
        if kind == "temp":
            return surface_temp_of_payload(payloads[-1], surf), ocean
        if kind == "temp_avg":
            fields = [surface_temp_of_payload(p, surf) for p in payloads[-AVG_WINDOW:]]
            return sum(fields) / len(fields), ocean
        if kind == "mld":
            return mld_of_payload(payloads[-1], maskT, zt)
        if kind == "mld_avg":
            pairs = [mld_of_payload(p, maskT, zt) for p in payloads[-AVG_WINDOW:]]
            valid = pairs[0][1]
            for _, wd in pairs[1:]:
                valid = valid & wd
            field = sum(m for m, _ in pairs) / len(pairs)
            return jnp.where(valid, field, 0.0), valid
        if kind == "tsdiff_avg":
            fields = [ts_diff_of_payload(p) for p in payloads[-AVG_WINDOW:]]
            return sum(fields) / len(fields), tsdiff_valid
        raise SystemExit(f"unknown observable {kind!r}")

    return observable
