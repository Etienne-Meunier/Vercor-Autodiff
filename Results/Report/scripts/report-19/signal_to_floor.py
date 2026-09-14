"""Report 19: signal-to-floor ratio for every candidate loss, vs rollout length.

Report-18 established that a 20-day identical-twin loss sits on a floor set by
chaotic decorrelation: perturbing c_eps by 1e-5 gives the same loss as perturbing
it by 2.5e-2. The whole optimization had only 46x of usable dynamic range and
stalled 3x above the floor. So the quantity that decides any future fit is

    signal / floor  =  loss(parameters wrong by a realistic amount)
                       -------------------------------------------
                       loss(parameters wrong by 1e-5)

measured per observable and per rollout length. This ranks every candidate on the
thing that actually binds, at the cost of 4 rollouts instead of a multi-hour fit.

Design: chained one-step couplers (as in report-17) to N_STEPS_MAX, so a single
rollout yields every checkpoint length AND the time-averaged observables, which
are the ones that should suppress the decorrelation noise (~1/sqrt(T)) while
keeping the parameter signal.

Observables, instantaneous and time-averaged:
  temp_surf      surface temperature
  dTdz_absavg3   mean |dT/dz| over the top 3 interfaces (report-18's loss)
  mld            mixed layer depth (report-16's loss)

Forward-only, no gradients.
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

import numpy as np
import jax.numpy as jnp

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import (
    JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig,
    make_jcm_land_atmosphere, make_veros_gcm,
)
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS, OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

OUT_DIR = os.environ.get("REPORT19_OUT_DIR", os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-19"))
os.makedirs(OUT_DIR, exist_ok=True)

N_MAX = int(os.environ.get("REPORT19_N_MAX", 30))
CHECKPOINTS = [int(x) for x in os.environ.get("REPORT19_CHECKPOINTS", "5,10,15,20,30").split(",")]
CHECKPOINTS = [n for n in CHECKPOINTS if n <= N_MAX]

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
POINTS = [
    ("target", TRUE_C_K, TRUE_C_EPS),
    ("floor", TRUE_C_K, TRUE_C_EPS + 1e-5),   # 0.0014% off: pure decorrelation noise
    ("ceps30", TRUE_C_K, 0.49),               # c_eps 30% low, where reports 15/18 stall
    ("start", 0.05, 0.40),                    # the actual optimization start
]

MLD_REF_DEPTH, MLD_REF_OFFSET = -10.0, 0.03
N_INTERFACES = 3


def get_index_mld(prho, maskT, zt, ref_depth, ref_offset=MLD_REF_OFFSET):
    level = jnp.arange(zt.shape[-1])
    ridx = jnp.max(jnp.where(zt < ref_depth, level, -1))
    prho_ref = prho[:, :, ridx] + ref_offset
    valid = maskT.astype(bool) & (level <= ridx)
    drho = prho - prho_ref[:, :, jnp.newaxis]
    below = valid & (drho > 0)
    has_below = jnp.any(below, axis=-1)
    i_below = jnp.argmax(jnp.where(below, zt, -jnp.inf), axis=-1)
    depth_below = zt[i_below]
    above = valid & (drho < 0) & (zt > depth_below[:, :, jnp.newaxis])
    has_above = jnp.any(above, axis=-1)
    i_above = jnp.argmin(jnp.where(above, zt, jnp.inf), axis=-1)
    return ridx, i_below, i_above, has_below & has_above


def mld_from_index(prho, zt, ridx, i_below, i_above, wd, ref_offset=MLD_REF_OFFSET):
    prho_ref = prho[:, :, ridx] + ref_offset
    pb = jnp.take_along_axis(prho, i_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    pa = jnp.take_along_axis(prho, i_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    db, da = zt[i_below], zt[i_above]
    denom = jnp.where(wd, pa - pb, 1.0)
    return jnp.where(wd, (prho_ref - pb) / denom * (da - db) + db, 0.0)


ocn = make_veros_gcm(config=VerosConfig(setup="global_4deg_learning", uses_atmosphere_forcing=True,
                                        restore_to_climatology=True, jitted=True, execution="jax"))
jcm = make_jcm_land_atmosphere(ocn.grid, config=JCMLandAtmosphereConfig(
    atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)))
lnd, atm = jcm.land, jcm.atmosphere
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)
START = datetime(2000, 1, 3)


def step_coupler(d):
    return Coupler(clock=Clock(start=d, dt_seconds=86400.0, steps=1, calendar="noleap"),
                   components=[ocn, lnd, atm], exchanges=exchanges, run_order=["OCN", "LND", "ATM"],
                   runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
                   log_level="WARNING")


print(f"building {N_MAX} one-step couplers ...", flush=True)
couplers = [step_coupler(START + timedelta(days=i)) for i in range(N_MAX)]
for c in couplers:
    c.initial_state()
init_state = couplers[0].initial_state()

_v = init_state._component_state("OCN").payload.variables
ZT = jnp.asarray(_v.zt)
MASKT = jnp.asarray(_v.maskT[2:-2, 2:-2, :]).astype(bool)
DZ_W = jnp.asarray(np.asarray(_v.zt)[1:] - np.asarray(_v.zt)[:-1])
MASK_NP = np.asarray(MASKT)
SURF = MASK_NP.shape[2] - 1
COLMASK = (MASK_NP[:, :, 1:] & MASK_NP[:, :, :-1])[:, :, -N_INTERFACES:].all(axis=2)


def observables(payload):
    v = payload.variables
    tau = v.tau
    temp = v.temp[2:-2, 2:-2, :, tau]
    prho = v.prho[2:-2, 2:-2, :]
    dTdz = (temp[:, :, 1:] - temp[:, :, :-1]) / DZ_W
    ridx, ib, ia, wd = get_index_mld(prho, MASKT, ZT, MLD_REF_DEPTH)
    mld = mld_from_index(prho, ZT, ridx, ib, ia, wd)
    return {
        "temp_surf": temp[:, :, SURF],
        "dTdz_absavg3": jnp.mean(jnp.abs(dTdz[:, :, -N_INTERFACES:]), axis=2),
        "mld": mld,
    }, wd


def rollout(c_k, c_eps):
    """Instantaneous and running-time-mean observables at every checkpoint."""
    cs = init_state._component_state("OCN")
    payload = set_veros_variable(cs.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = init_state._with_component_state("OCN", cs.with_payload(payload))

    sums, wd_all, out = None, None, {}
    for step, cpl in enumerate(couplers, start=1):
        state = cpl.run(state, output=None)
        obs, wd = observables(state._component_state("OCN").payload)
        sums = obs if sums is None else {k: sums[k] + obs[k] for k in obs}
        wd_all = wd if wd_all is None else (wd_all & wd)
        if step in CHECKPOINTS:
            for k in obs:
                out[f"{k}@{step}"] = obs[k]
                out[f"{k}_tmean@{step}"] = sums[k] / step
            out[f"wd@{step}"] = wd_all.astype(jnp.float64)
    return out


rollout_jit = jax.jit(rollout)

results = {}
for name, a, b in POINTS:
    t0 = time.time()
    results[name] = jax.tree.map(np.asarray, rollout_jit(jnp.asarray(a), jnp.asarray(b)))
    print(f"  {name:>7s} (c_k={a}, c_eps={b}): {time.time() - t0:.1f}s", flush=True)

OBS = ["temp_surf", "dTdz_absavg3", "mld"]
rows = []
for n in CHECKPOINTS:
    wd = results["target"][f"wd@{n}"].astype(bool)
    for base in OBS:
        for kind in ("", "_tmean"):
            key = f"{base}{kind}@{n}"
            m = wd if base == "mld" else (COLMASK if base == "dTdz_absavg3" else MASK_NP[:, :, SURF])
            tgt = results["target"][key]

            def loss(pt):
                d = np.where(m, results[pt][key] - tgt, 0.0)
                return float(np.sum(d ** 2))

            f, s30, sst = loss("floor"), loss("ceps30"), loss("start")
            rows.append((n, f"{base}{kind}", int(m.sum()), f, s30, sst,
                         s30 / f if f > 0 else np.inf, sst / f if f > 0 else np.inf))

np.save(os.path.join(OUT_DIR, "signal_to_floor.npy"), np.array(rows, dtype=object), allow_pickle=True)

print(f"\n{'N':>3s} {'observable':>20s} {'cells':>6s} {'floor':>11s} {'sig(ceps30)':>12s} "
      f"{'S/F ceps':>9s} {'S/F start':>10s}", flush=True)
for r in rows:
    print(f"{r[0]:3d} {r[1]:>20s} {r[2]:6d} {r[3]:11.3e} {r[4]:12.3e} {r[6]:9.1f} {r[7]:10.1f}", flush=True)

print("\nS/F ceps = how many times louder a 30% c_eps error is than pure decorrelation noise.", flush=True)
print("Report-18's N=20 mean(|dT/dz|) loss had S/F(start) = 46 and stalled 3x above its floor.", flush=True)
print("done.", flush=True)
