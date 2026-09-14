"""Report 8 addendum: cheap-fix follow-up to landscape_multistart.py's
overshoot-then-recover hump (all 3 cosine-schedule starts there peak at
17-24x their eventual best loss around iteration 3-9, before decaying
back down). Halves the cosine schedule's peak lr (2e-2 -> 1e-2), same
N_ITERS=30, same 3 starts, to see whether a smaller peak step size tames
the hump within the same iteration budget.

Does NOT redo the grid scan (unneeded for this comparison) -- only the 3
optimizations, with new checkpoint names (start_{i}_lowlr_trajectory.npy)
so the existing lr=2e-2 results in start_{i}_trajectory.npy are left
untouched for the loss-vs-iteration comparison plot.
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

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax

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

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-8")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 20
N_ITERS = 30
LEARNING_RATE = 1e-2  # half of landscape_multistart.py's 2e-2
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7

STARTS = [
    (0.05, 0.40),
    (0.15, 0.85),
    (0.03, 0.55),
]

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
)
initial_state = cpl.initial_state()


def final_ocn_temp(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload.variables.temp


print(f"Generating target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...")
target_temp = jax.jit(final_ocn_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_temp = jax.lax.stop_gradient(target_temp)


def loss_fn(params):
    temp = final_ocn_temp(params["c_k"], params["c_eps"])
    return jnp.sum((temp - target_temp) ** 2)


def run_optimization(init_ck, init_ceps, checkpoint_path):
    if os.path.exists(checkpoint_path):
        traj = [tuple(row) for row in np.load(checkpoint_path).tolist()]
        print(f"resuming optimization from checkpoint ({len(traj) - 1} / {N_ITERS} iters already done)", flush=True)
    else:
        traj = [(init_ck, init_ceps, None)]

    params = {"c_k": jnp.asarray(traj[-1][0]), "c_eps": jnp.asarray(traj[-1][1])}
    schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    for iteration in range(len(traj) - 1, N_ITERS):
        loss_value, grads = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
        traj.append((ck_val, ceps_val, float(loss_value)))
        print(f"  iter {iteration:3d}  loss={float(loss_value):.4e}  c_k={ck_val:.4f}  c_eps={ceps_val:.4f}", flush=True)
        np.save(checkpoint_path, np.array([[t[0], t[1], t[2] if t[2] is not None else np.nan] for t in traj]))
    return traj


lowlr_trajectories = []
for idx, (start_ck, start_ceps) in enumerate(STARTS):
    print(f"\nOptimizing from start (c_k={start_ck}, c_eps={start_ceps}), lr={LEARNING_RATE} ...", flush=True)
    ckpt_path = os.path.join(OUT_DIR, f"start_{idx}_lowlr_trajectory.npy")
    lowlr_trajectories.append(run_optimization(start_ck, start_ceps, ckpt_path))

# --- comparison figure: loss vs iteration, lr=2e-2 (existing) vs lr=1e-2 (this run) ---
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (idx, (sck, sce)) in zip(axs, enumerate(STARTS)):
    hi_path = os.path.join(OUT_DIR, f"start_{idx}_trajectory.npy")
    hi_traj = np.load(hi_path)
    hi_losses = hi_traj[1:, 2]
    lo_losses = np.array([t[2] for t in lowlr_trajectories[idx][1:]])

    ax.plot(range(len(hi_losses)), hi_losses, "-o", color="tab:red", markersize=3, label="lr=2e-2 (peak)")
    ax.plot(range(len(lo_losses)), lo_losses, "-o", color="tab:blue", markersize=3, label="lr=1e-2 (peak)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"start ({sck}, {sce})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    hi_best = float(np.min(hi_losses))
    lo_best = float(np.min(lo_losses))
    print(f"start ({sck},{sce}): lr=2e-2 best={hi_best:.4e}  lr=1e-2 best={lo_best:.4e}")

fig.suptitle("Report 8: peak-lr comparison (cosine decay, N_ITERS=30) -- does a smaller peak lr tame the overshoot hump?")
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_path = os.path.join(OUT_DIR, "landscape_lr_comparison.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nsaved {out_path}")
