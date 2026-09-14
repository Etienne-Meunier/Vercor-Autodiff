"""Report 17: forward-mode sensitivity and identifiability maps for c_k / c_eps.

Reports 15 and 16 established that a 20-day surface-temperature loss cannot
identify `c_eps` while an MLD loss can. Both conclusions came from running full
optimizations. This script answers the same question directly, and much more
cheaply, with forward-mode AD.

Method
------
The rollout is chained from N_STEPS one-step couplers (same construction as
report-6's windowed runs) so that a *single* jvp yields the parameter tangent of
every diagnostic field at every step. Forward mode is the right mode here: two
parameters, many outputs, and no tape to store, so cost is ~2 forward runs for
the whole space-time-field sensitivity tensor.

For each evaluation point (c_k, c_eps) we compute two tangents,

    J_k = d(field) / d(c_k)      via jvp with tangent (1, 0)
    J_e = d(field) / d(c_eps)    via jvp with tangent (0, 1)

and work throughout with the log-sensitivities

    S_k = c_k   * J_k = d(field) / d(log c_k)
    S_e = c_eps * J_e = d(field) / d(log c_eps)

which put both parameters in fractional units and so make the comparison between
them meaningful.

Identifiability
---------------
Sensitivity magnitude is not the question reports 15/16 ran into -- separability
is. For each field, over its valid cell set M, we form the 2x2 Gauss-Newton
(Gram) matrix of the log-sensitivities,

    H = [[<S_k,S_k>, <S_k,S_e>],
         [<S_k,S_e>, <S_e,S_e>]]

    r    = <S_k,S_e> / (|S_k| |S_e|)     collinearity, in [-1, 1]
    cond = lambda_max / lambda_min       identifiability (1 = perfectly separable)

A weighted least-squares loss on that field has exactly this H as its
Gauss-Newton Hessian at the evaluation point, so |r| -> 1 or a large cond means a
flat valley in parameter space that no optimizer or learning-rate schedule can
fix -- precisely report-8's flat `c_eps` band. Fields are compared on a
dimensionless relative sensitivity, rms(S) / std(field over M), so that a field
in deg C and a field in 1/s^2 can be ranked side by side.

Hypothesis under test
---------------------
Report-16 found MLD informative about `c_eps` while surface temperature is not,
even though MLD is diagnosed from a density field dominated by temperature. The
proposed explanation is that MLD reads the *vertical structure* of the density
profile, not its values: what mixing sets is d(rho)/dz, and the MLD threshold
crossing is a (discontinuous) readout of it. If that is right, then `Nsqr` and
dT/dz should show a `c_eps` sensitivity and a separability comparable to MLD's
and much better than temperature's -- and, being smooth, would give the MLD
signal without report-16's piecewise-constant level-flip jumps.

Outputs
-------
Figures and .npy dumps in Results/Report/figures/report-17/.

Env overrides (smoke test): REPORT17_N_STEPS, REPORT17_OUT_DIR, REPORT17_POINTS.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT_DIR = os.environ.get(
    "REPORT17_OUT_DIR", os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-17")
)
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = int(os.environ.get("REPORT17_N_STEPS", 20))
START_DATE = datetime(2000, 1, 3, 0, 0, 0)

MLD_REFERENCE_DEPTH = -10.0
MLD_REFERENCE_OFFSET = 0.03

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

# Evaluation points. Local tangents only describe the neighbourhood they are taken
# in, and report-8's low-loss band is wide, so the same analysis is repeated at
# the truth, at reports 15/16's starting guess, and at the point report-15's
# c_eps stalled on.
ALL_POINTS = [
    ("truth", TRUE_C_K, TRUE_C_EPS),
    ("init", 0.05, 0.40),
    ("stall", 0.0876, 0.4892),
]
_pts = os.environ.get("REPORT17_POINTS")
POINTS = [p for p in ALL_POINTS if p[0] in _pts.split(",")] if _pts else ALL_POINTS


# --- MLD diagnostic (same kernel as report-16) --------------------------------

def get_index_mld(prho, maskT, zt, reference_depth, reference_offset=MLD_REFERENCE_OFFSET):
    """Level indices bracketing the MLD -- discrete selection only, no gradient path."""
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
    """MLD from precomputed level indices -- plain gather + arithmetic, differentiable."""
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below = jnp.take_along_axis(prho, i_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    prho_above = jnp.take_along_axis(prho, i_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    depth_below = zt[i_below]
    depth_above = zt[i_above]

    denom = jnp.where(well_defined, prho_above - prho_below, 1.0)
    mld = (prho_reference - prho_below) / denom * (depth_above - depth_below) + depth_below
    return jnp.where(well_defined, mld, 0.0)


# --- model construction -------------------------------------------------------

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

exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def make_step_coupler(start_date):
    """One-step coupler. Chaining N_STEPS of these is equivalent to a single
    N_STEPS run (same construction report-6 uses for windowed rollouts) and is
    what makes the per-step tangent fall out of one jvp."""
    clock = Clock(start=start_date, dt_seconds=86400.0, steps=1, calendar="noleap")
    return Coupler(
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
        log_level="WARNING",  # per-step INFO logging fires a host callback per step
    )


print(f"building {N_STEPS} one-step couplers ...", flush=True)
_t0 = time.time()
step_couplers = [make_step_coupler(START_DATE + timedelta(days=i)) for i in range(N_STEPS)]
for cpl in step_couplers:
    cpl.initial_state()
initial_state = step_couplers[0].initial_state()
print(f"  done ({time.time() - _t0:.1f}s)", flush=True)

_init_vars = initial_state._component_state("OCN").payload.variables
ZT = jnp.asarray(_init_vars.zt)
MASKT = jnp.asarray(_init_vars.maskT[2:-2, 2:-2, :]).astype(bool)
ZT_NP = np.asarray(_init_vars.zt)
NZ = ZT_NP.size
SURFACE = NZ - 1  # zt is ordered deepest-first; zt[-1] is the surface cell centre
XT = np.asarray(ocn.grid.longitude)
YT = np.asarray(ocn.grid.latitude)
DZ_W = jnp.asarray(ZT_NP[1:] - ZT_NP[:-1])  # spacing between adjacent cell centres

print(f"zt (m): {np.round(ZT_NP, 1).tolist()}", flush=True)


# --- diagnostics --------------------------------------------------------------
# Every entry is a float array so the whole dict can be a jvp output. 3D fields
# are (nx, ny, nz); interface fields are (nx, ny, nz-1); mld is (nx, ny).

FIELD_KIND = {
    "temp": "T",
    "salt": "T",
    "prho": "T",
    "Nsqr": "T",
    "tke": "T",
    "dTdz": "W",
    "dSdz": "W",
    "drhodz": "W",
    "mld": "S",
}


def diagnostics(payload):
    v = payload.variables
    tau = v.tau
    temp = v.temp[2:-2, 2:-2, :, tau]
    salt = v.salt[2:-2, 2:-2, :, tau]
    prho = v.prho[2:-2, 2:-2, :]
    nsqr = v.Nsqr[2:-2, 2:-2, :, tau]
    tke = v.tke[2:-2, 2:-2, :, tau]

    ridx, i_below, i_above, well_defined = get_index_mld(prho, MASKT, ZT, MLD_REFERENCE_DEPTH)
    mld = mld_from_index(prho, ZT, ridx, i_below, i_above, well_defined)

    return {
        "temp": temp,
        "salt": salt,
        "prho": prho,
        "Nsqr": nsqr,
        "tke": tke,
        "dTdz": (temp[:, :, 1:] - temp[:, :, :-1]) / DZ_W,
        "dSdz": (salt[:, :, 1:] - salt[:, :, :-1]) / DZ_W,
        "drhodz": (prho[:, :, 1:] - prho[:, :, :-1]) / DZ_W,
        "mld": mld,
    }


def rollout(c_k, c_eps):
    """Chained N_STEPS rollout returning every diagnostic at every step, stacked
    on a leading time axis."""
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))

    per_step = []
    for cpl in step_couplers:
        state = cpl.run(state, output=None)
        per_step.append(diagnostics(state._component_state("OCN").payload))

    return {k: jnp.stack([d[k] for d in per_step], axis=0) for k in per_step[0]}


@jax.jit
def jvp_rollout(c_k, c_eps, t_k, t_eps):
    return jax.jvp(rollout, (c_k, c_eps), (t_k, t_eps))


def well_defined_mask(c_k, c_eps):
    """MLD validity mask of the final state -- boolean, so kept out of the jvp."""
    component_state = initial_state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", jnp.asarray(c_k))
    payload = set_veros_variable(payload, "c_eps", jnp.asarray(c_eps))
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    for cpl in step_couplers:
        state = cpl.run(state, output=None)
    prho = state._component_state("OCN").payload.variables.prho[2:-2, 2:-2, :]
    return get_index_mld(prho, MASKT, ZT, MLD_REFERENCE_DEPTH)[3]


# --- run the tangents ---------------------------------------------------------

ONE, ZERO = jnp.asarray(1.0), jnp.asarray(0.0)
results = {}

for name, ck, ce in POINTS:
    print(f"\n=== point '{name}': c_k={ck}, c_eps={ce} ===", flush=True)
    ck_j, ce_j = jnp.asarray(ck), jnp.asarray(ce)

    t0 = time.time()
    primal, tan_k = jvp_rollout(ck_j, ce_j, ONE, ZERO)
    primal = jax.tree.map(lambda x: np.asarray(x), primal)
    tan_k = jax.tree.map(lambda x: np.asarray(x), tan_k)
    print(f"  d/dc_k    jvp: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    _, tan_e = jvp_rollout(ck_j, ce_j, ZERO, ONE)
    tan_e = jax.tree.map(lambda x: np.asarray(x), tan_e)
    print(f"  d/dc_eps  jvp: {time.time() - t0:.1f}s", flush=True)

    wd = np.asarray(well_defined_mask(ck, ce))

    n_nan = sum(int(np.isnan(v).sum()) for v in list(tan_k.values()) + list(tan_e.values()))
    print(f"  NaNs in tangents: {n_nan}", flush=True)

    # Log-sensitivities: S = theta * d(field)/d(theta) = d(field)/d(log theta).
    results[name] = {
        "c_k": ck,
        "c_eps": ce,
        "primal": primal,
        "S_k": {k: ck * v for k, v in tan_k.items()},
        "S_e": {k: ce * v for k, v in tan_e.items()},
        "mld_mask": wd,
    }

np.save(os.path.join(OUT_DIR, "zt.npy"), ZT_NP)
np.save(os.path.join(OUT_DIR, "xt.npy"), XT)
np.save(os.path.join(OUT_DIR, "yt.npy"), YT)
np.save(os.path.join(OUT_DIR, "maskT.npy"), np.asarray(MASKT))

# Final-step fields, per point -- enough for figures.py to regenerate every map
# without repeating the rollouts.
for name, res in results.items():
    for key in ("S_k", "S_e", "primal"):
        for field, arr in res[key].items():
            np.save(os.path.join(OUT_DIR, f"{name}_{key}_{field}_final.npy"), arr[-1])
    np.save(os.path.join(OUT_DIR, f"{name}_mld_mask.npy"), res["mld_mask"])


# --- masks and Gram analysis --------------------------------------------------

MASK_NP = np.asarray(MASKT)
MASK_W = MASK_NP[:, :, 1:] & MASK_NP[:, :, :-1]


def field_mask(field, mld_mask):
    kind = FIELD_KIND[field]
    if kind == "T":
        return MASK_NP
    if kind == "W":
        return MASK_W
    return mld_mask  # "S": MLD, valid columns only


def gram(sk, se, mask):
    """2x2 Gauss-Newton matrix of the log-sensitivities over `mask`, plus the
    derived collinearity, condition number and weak eigendirection."""
    a = float(np.sum(sk[mask] ** 2))
    b = float(np.sum(sk[mask] * se[mask]))
    c = float(np.sum(se[mask] ** 2))
    tr, det = a + c, a * c - b * b
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    lmax, lmin = tr / 2.0 + disc, tr / 2.0 - disc
    r = b / (a * c) ** 0.5 if a > 0 and c > 0 else float("nan")
    cond = lmax / lmin if lmin > 0 else float("inf")
    # eigenvector for lmin (the weak / poorly-determined direction)
    if abs(b) > 0:
        v = np.array([b, lmin - a])
    else:
        v = np.array([0.0, 1.0]) if a >= c else np.array([1.0, 0.0])
    v = v / (np.linalg.norm(v) + 1e-300)
    return dict(a=a, b=b, c=c, lmax=lmax, lmin=lmin, r=r, cond=cond, v_min=v)


def rel_sens(s, y, mask):
    """rms(S) / std(field): dimensionless, so fields in different units compare."""
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    sd = float(np.std(y[mask]))
    if sd == 0:
        return float("nan")
    return float(np.sqrt(np.sum(s[mask] ** 2) / n) / sd)


FIELDS = ["temp", "salt", "prho", "Nsqr", "tke", "dTdz", "dSdz", "drhodz", "mld"]

summary = {}
for name, res in results.items():
    rows = {}
    for field in FIELDS:
        mask = field_mask(field, res["mld_mask"])
        sk, se = res["S_k"][field][-1], res["S_e"][field][-1]
        y = res["primal"][field][-1]
        g = gram(sk, se, mask)
        g["rel_k"] = rel_sens(sk, y, mask)
        g["rel_e"] = rel_sens(se, y, mask)
        g["n"] = int(mask.sum())
        rows[field] = g
    # surface level of the 3D fields, treated as its own observable (report-15's
    # loss domain).
    for field in ("temp", "salt", "prho", "Nsqr", "tke"):
        mask = MASK_NP[:, :, SURFACE]
        sk, se = res["S_k"][field][-1, :, :, SURFACE], res["S_e"][field][-1, :, :, SURFACE]
        y = res["primal"][field][-1, :, :, SURFACE]
        g = gram(sk, se, mask)
        g["rel_k"] = rel_sens(sk, y, mask)
        g["rel_e"] = rel_sens(se, y, mask)
        g["n"] = int(mask.sum())
        rows[f"{field}_surf"] = g
    summary[name] = rows

ORDER = [
    "temp_surf", "temp", "salt_surf", "salt", "prho_surf", "prho",
    "mld", "dTdz", "dSdz", "drhodz", "Nsqr_surf", "Nsqr", "tke_surf", "tke",
]

print("\n" + "=" * 100, flush=True)
for name in summary:
    print(f"\nfinal-step identifiability at '{name}' (c_k={results[name]['c_k']}, c_eps={results[name]['c_eps']})", flush=True)
    print(f"{'field':>10s} {'cells':>7s} {'rel_k':>10s} {'rel_e':>10s} {'rel_e/rel_k':>12s} {'r':>8s} {'cond':>12s}", flush=True)
    for field in ORDER:
        g = summary[name][field]
        ratio = g["rel_e"] / g["rel_k"] if g["rel_k"] else float("nan")
        print(f"{field:>10s} {g['n']:7d} {g['rel_k']:10.3e} {g['rel_e']:10.3e} {ratio:12.3e} "
              f"{g['r']:8.4f} {g['cond']:12.4e}", flush=True)

np.save(os.path.join(OUT_DIR, "summary.npy"), np.array([summary], dtype=object), allow_pickle=True)


# --- time series --------------------------------------------------------------

TS_FIELDS = ["temp_surf", "temp", "salt", "mld", "dTdz", "Nsqr", "tke"]


def slice_field(res, key, field, t):
    if field.endswith("_surf"):
        return res[key][field[:-5]][t, :, :, SURFACE]
    return res[key][field][t]


def ts_mask(res, field):
    if field.endswith("_surf"):
        return MASK_NP[:, :, SURFACE]
    return field_mask(field, res["mld_mask"])


timeseries = {}
for name, res in results.items():
    ts = {f: {"rel_k": [], "rel_e": [], "r": [], "cond": []} for f in TS_FIELDS}
    for field in TS_FIELDS:
        mask = ts_mask(res, field)
        for t in range(N_STEPS):
            sk = slice_field(res, "S_k", field, t)
            se = slice_field(res, "S_e", field, t)
            y = (res["primal"][field[:-5]][t, :, :, SURFACE] if field.endswith("_surf")
                 else res["primal"][field][t])
            g = gram(sk, se, mask)
            ts[field]["rel_k"].append(rel_sens(sk, y, mask))
            ts[field]["rel_e"].append(rel_sens(se, y, mask))
            ts[field]["r"].append(g["r"])
            ts[field]["cond"].append(g["cond"])
    timeseries[name] = {f: {k: np.array(v) for k, v in d.items()} for f, d in ts.items()}

np.save(os.path.join(OUT_DIR, "timeseries.npy"), np.array([timeseries], dtype=object), allow_pickle=True)


# --- level-by-level breakdown (needed by the figures, cheap to compute here) ---

REF = POINTS[0][0]
res = results[REF]
LVL_FIELDS = ["temp", "salt", "prho", "Nsqr", "tke"]
rel_e_lvl = np.full((len(LVL_FIELDS), NZ), np.nan)
r_lvl = np.full((len(LVL_FIELDS), NZ), np.nan)
for i, field in enumerate(LVL_FIELDS):
    for k in range(NZ):
        m = MASK_NP[:, :, k]
        if m.sum() < 10:
            continue
        sk = res["S_k"][field][-1, :, :, k]
        se = res["S_e"][field][-1, :, :, k]
        y = res["primal"][field][-1, :, :, k]
        rel_e_lvl[i, k] = rel_sens(se, y, m)
        r_lvl[i, k] = abs(gram(sk, se, m)["r"])

np.save(
    os.path.join(OUT_DIR, "meta.npy"),
    np.array([{
        "points": [p[0] for p in POINTS],
        "point_params": {p[0]: (p[1], p[2]) for p in POINTS},
        "n_steps": N_STEPS,
        "lvl_fields": LVL_FIELDS,
        "rel_e_lvl": rel_e_lvl,
        "r_lvl": r_lvl,
    }], dtype=object),
    allow_pickle=True,
)


# --- figures ------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures import make_figures  # noqa: E402

make_figures(OUT_DIR)

print("\ndone.", flush=True)
