"""Shared model setup and stratification loss for report 18.

The observable is the report-17 stratification metric: the mean of |dT/dz| over
the top 3 interfaces of each column, i.e. one number per surface cell.

    strat(x, y) = mean over the 3 shallowest interfaces of |(T[k+1]-T[k]) / (z[k+1]-z[k])|

Report-17 measured this as the strongest *smooth* observable for identifying
`c_eps`: 62x surface temperature's curvature along the weak parameter direction,
stable across all three evaluation points, and with no discrete level selection
anywhere in it (unlike MLD, whose loss is piecewise-smooth with jumps where the
bracketing level pair flips).

`jnp.abs` is not differentiable at 0; JAX takes the subgradient sign(0) = 0
there. No column in this setup sits at exactly zero gradient.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

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

# Recipe: identical to report-15 / report-16 so the only change is the observable.
N_STEPS = int(os.environ.get("REPORT18_N_STEPS", 20))
N_ITERS = int(os.environ.get("REPORT18_N_ITERS", 30))
LEARNING_RATE = 2e-2  # peak lr of the cosine-decay schedule

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4

N_INTERFACES = 3  # the metric averages over the top 3 interfaces


def build():
    """Coupler, initial state, and the grid constants the metric needs."""
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
    lnd, atm = jcm_setup.land, jcm_setup.atmosphere

    clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=N_STEPS, calendar="noleap")
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
        log_level="WARNING",  # per-step INFO logging fires a host callback per step
    )
    initial_state = cpl.initial_state()

    v = initial_state._component_state("OCN").payload.variables
    zt = np.asarray(v.zt)
    maskT = np.asarray(v.maskT[2:-2, 2:-2, :]).astype(bool)

    # Interface k sits between cell centres k and k+1; levels are ordered
    # deepest-first, so the top N interfaces are the last N.
    mask_w = maskT[:, :, 1:] & maskT[:, :, :-1]
    column_mask = mask_w[:, :, -N_INTERFACES:].all(axis=2)

    grid = {
        "zt": zt,
        "maskT": maskT,
        "column_mask": column_mask,
        "dz_w": jnp.asarray(zt[1:] - zt[:-1]),
        "column_mask_j": jnp.asarray(column_mask),
        "longitude": np.asarray(ocn.grid.longitude),
        "latitude": np.asarray(ocn.grid.latitude),
    }
    return cpl, initial_state, grid


def make_kernels(cpl, initial_state, grid):
    """(strat_of_params, loss_factory) closed over the built model."""

    dz_w = grid["dz_w"]
    column_mask = grid["column_mask_j"]

    def final_payload(c_k, c_eps):
        component_state = initial_state._component_state("OCN")
        payload = set_veros_variable(component_state.payload, "c_k", c_k)
        payload = set_veros_variable(payload, "c_eps", c_eps)
        state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
        return cpl.run(state, output=None)._component_state("OCN").payload

    def strat_of_payload(payload):
        tau = payload.variables.tau
        temp = payload.variables.temp[2:-2, 2:-2, :, tau]
        dTdz = (temp[:, :, 1:] - temp[:, :, :-1]) / dz_w
        return jnp.mean(jnp.abs(dTdz[:, :, -N_INTERFACES:]), axis=2)

    def strat(c_k, c_eps):
        return strat_of_payload(final_payload(c_k, c_eps))

    def make_loss(target):
        def loss_fn(params):
            s = strat(params["c_k"], params["c_eps"])
            diff = jnp.where(column_mask, s - target, 0.0)
            return jnp.sum(diff ** 2)
        return loss_fn

    return strat, make_loss


def best_iterate(traj):
    """Pair each recorded loss with the parameters it was actually evaluated at.

    Checkpoint row k+1 holds the parameters produced *after* iteration k's update
    alongside the loss evaluated *before* it, i.e. at row k's parameters. Report-16
    documented this off-by-one; pairing naively names parameters one adam step past
    the ones that were scored.
    """
    best_i, best_l = -1, float("inf")
    best_p = (traj[0][0], traj[0][1])
    for k in range(len(traj) - 1):
        loss = traj[k + 1][2]
        if loss is not None and not np.isnan(loss) and loss < best_l:
            best_l, best_i = loss, k
            best_p = (traj[k][0], traj[k][1])
    return best_i, best_l, best_p[0], best_p[1]


def load_trajectory(path, n_iters, start):
    """Load a checkpointed trajectory, but ONLY if it is complete.

    Resuming a partial run is scientifically wrong here: `optimizer.init` rebuilds
    adam's first/second-moment estimates from zero, so a run resumed at iteration k
    is not the same recipe as an uninterrupted one, and is not comparable to
    report-15/16. A 2026-09-07 g5k run hit exactly this -- two jobs resumed each
    other's partial checkpoint and produced visibly different trajectories from the
    same start (c_eps 0.905 vs 0.915).

    So: a COMPLETE trajectory is reused (nothing to recompute), a PARTIAL one is
    discarded and the run restarts from scratch. That keeps crash-tolerance at the
    granularity of a whole optimization while never silently reporting a
    moment-reset trajectory as if it were a clean one.
    """
    if not os.path.exists(path):
        return [(start[0], start[1], None)], False
    traj = [tuple(r) for r in np.load(path).tolist()]
    if len(traj) - 1 >= n_iters:
        print(f"  reusing COMPLETE trajectory from {os.path.basename(path)} "
              f"({len(traj) - 1} iterations)", flush=True)
        return traj, True
    print(f"  discarding PARTIAL trajectory in {os.path.basename(path)} "
          f"({len(traj) - 1}/{n_iters} iters): resuming would reset adam's moment "
          f"estimates and break comparability. Restarting this optimization from scratch.",
          flush=True)
    return [(start[0], start[1], None)], False


def acquire_lock(out_dir):
    """Refuse to run if another job is already writing this output directory.

    Three duplicate OAR jobs once raced on the same checkpoints; this makes that
    a loud failure instead of silent corruption.
    """
    lock = os.path.join(out_dir, ".job.lock")
    me = os.environ.get("OAR_JOB_ID", f"pid-{os.getpid()}")
    if os.path.exists(lock):
        holder = open(lock).read().strip()
        if holder != me:
            raise SystemExit(
                f"ERROR: {lock} is held by '{holder}'. Another job is writing this "
                f"directory. Kill it (or remove the lock file) before rerunning."
            )
    with open(lock, "w") as f:
        f.write(me)
    return lock


def release_lock(lock):
    try:
        os.remove(lock)
    except OSError:
        pass
