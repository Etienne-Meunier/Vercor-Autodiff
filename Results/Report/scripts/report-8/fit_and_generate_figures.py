"""Report 8: c_k/c_eps fit at N_STEPS=20, "classical" direct-gradient recipe
(single value_and_grad call over the whole rollout, no windowing/teacher
forcing) -- a clean worked example that classical gradient-based
calibration works on the fully coupled OCN+LND+ATM(jcm) model, at the
largest rollout length report-3 found gradients still trustworthy at
(n=20, ~7-8% FD rel err there).

v2 (this file): report-3/first version of this report used a *fixed*
optax.adam learning rate, which overshoot the true minimum (loss dips to
0.877 at iteration 17, then rises back to 2.75 by iteration 19 -- the
run never settles). This version keeps optax.adam (still the right
choice here: no top-level jax.jit of the whole coupled rollout exists in
this codebase, so every extra loss/grad evaluation costs a full
~130-150s wall-clock forward+backward pass on CPU -- ruling out
line-search optimizers like L-BFGS, which need several extra evaluations
per step) but adds a cosine-decay learning-rate schedule so step size
shrinks to ~0 by the end of the run instead of staying fixed. This is
the standard fix for exactly this failure mode and costs nothing extra
per iteration (schedule value is closed-form, no added evaluations).
N_ITERS is also raised 20 -> 30 for a longer, fixed-length run (no early
stop: the loss trajectory has a genuine overshoot-then-recover hump, and
a "no improvement over best" stopping rule was tried and fires mid-hump,
cutting the run off before the recovery -- worse than just running the
full 30 iterations). The "best iteration" (lowest loss seen, not just
the last one) is tracked and used for the optimized-field snapshot,
since report-8 v1 showed the final iteration is not always the best.

Three figures now: trajectory (loss + (c_k, c_eps) path, best iteration
marked), and a temperature-bias snapshot across THREE fitted parameter
states (init / best / final) against the target, instead of two.

Script: `Results/Report/scripts/report-8/fit_and_generate_figures.py`.
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

N_STEPS = 20  # largest rollout report-3 found gradients still trustworthy at
N_ITERS = 30  # raised from 20 (v1), full fixed-length run (no early stop --
# the loss trajectory has a genuine overshoot-then-recover hump, so a
# "no improvement over best" stopping rule fires mid-hump and cuts off
# before the recovery, as a first attempt at this script confirmed)
LEARNING_RATE = 2e-2  # peak lr of the cosine-decay schedule (v1's fixed lr)

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4
Z_LEVEL = -1  # surface level

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


def final_ocn_payload(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload


def final_ocn_temp(c_k, c_eps):
    return final_ocn_payload(c_k, c_eps).variables.temp


print(f"Generating synthetic target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...")
target_temp = jax.jit(final_ocn_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_temp = jax.lax.stop_gradient(target_temp)


def loss_fn(params):
    temp = final_ocn_temp(params["c_k"], params["c_eps"])
    return jnp.sum((temp - target_temp) ** 2)


params = {"c_k": jnp.asarray(INIT_C_K), "c_eps": jnp.asarray(INIT_C_EPS)}
schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)
value_and_grad_fn = jax.value_and_grad(loss_fn)

trajectory = [(INIT_C_K, INIT_C_EPS, None)]
best_iter, best_loss = -1, float("inf")
best_ck, best_ceps = INIT_C_K, INIT_C_EPS
iter_wall_times = []

print(f"\n{'iter':>4s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}  {'lr':>10s}  {'wall_s':>8s}")
print(f"{'init':>4s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")

t_start = time.time()
for iteration in range(N_ITERS):
    t0 = time.time()
    loss_value, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    dt = time.time() - t0
    iter_wall_times.append(dt)

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    loss_val = float(loss_value)
    trajectory.append((ck_val, ceps_val, loss_val))
    print(f"{iteration:4d}  {loss_val:14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  {float(schedule(iteration)):10.4f}  {dt:8.1f}")

    if loss_val < best_loss:
        best_loss, best_iter = loss_val, iteration
        best_ck, best_ceps = ck_val, ceps_val

total_wall = time.time() - t_start
FINAL_C_K, FINAL_C_EPS = trajectory[-1][0], trajectory[-1][1]
FINAL_LOSS = trajectory[-1][2]
print(f"\ntotal descent wall time: {total_wall:.1f}s over {len(iter_wall_times)} iterations "
      f"(mean {np.mean(iter_wall_times):.1f}s/iter)")
print("recovered c_k (final):  ", FINAL_C_K, " (true:", TRUE_C_K, ")")
print("recovered c_eps (final):", FINAL_C_EPS, " (true:", TRUE_C_EPS, ")")
print(f"best iteration: {best_iter}, loss={best_loss:.6e}, c_k={best_ck:.4f}, c_eps={best_ceps:.4f}")

# --- snapshot rollouts: target, init, best, final (three fitted parameter states) ---
print("\nrolling out initial guess for snapshot ...")
initial_payload = jax.jit(final_ocn_payload)(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS))
print("rolling out best-iteration params for snapshot ...")
best_payload = jax.jit(final_ocn_payload)(jnp.asarray(best_ck), jnp.asarray(best_ceps))
print("rolling out final params for snapshot ...")
final_payload = jax.jit(final_ocn_payload)(jnp.asarray(FINAL_C_K), jnp.asarray(FINAL_C_EPS))
target_payload_full = jax.jit(final_ocn_payload)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))


def field_at_level(payload, z):
    tau = payload.variables.tau
    temp = np.asarray(payload.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(payload.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


xt = np.asarray(ocn.grid.longitude)
yt = np.asarray(ocn.grid.latitude)

target_field = field_at_level(target_payload_full, Z_LEVEL)
bias_initial = field_at_level(initial_payload, Z_LEVEL) - target_field
bias_best = field_at_level(best_payload, Z_LEVEL) - target_field
bias_final = field_at_level(final_payload, Z_LEVEL) - target_field

bias_max = np.nanmax(
    np.abs(
        np.concatenate([
            bias_initial[~np.isnan(bias_initial)],
            bias_best[~np.isnan(bias_best)],
            bias_final[~np.isnan(bias_final)],
        ])
    )
)

# --- figure 1: trajectory ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

iters = list(range(len(trajectory) - 1))
losses = [t[2] for t in trajectory[1:]]
axs[0].plot(iters, losses, "o-", color="tab:green")
axs[0].axvline(best_iter, color="tab:orange", linestyle="--", alpha=0.7, label=f"best (iter {best_iter})")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared temp error)")
axs[0].set_yscale("log")
axs[0].set_title("Loss vs iteration (log scale)")
axs[0].legend()
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="tab:green", markersize=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(best_ck, best_ceps, "D", color="tab:orange", markersize=9, label="best", zorder=4)
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:green", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 8: c_k/c_eps recovery, {N_STEPS}-step coupled OCN+LND+ATM(jcm) rollout, "
             f"adam + cosine-decay lr")
fig.tight_layout()
traj_path = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"saved {traj_path}")

# --- figure 2: temperature bias snapshot, three fitted parameter states ---
fig, axs = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[0].set_title(f"target (surface, c_k={TRUE_C_K}, c_eps={TRUE_C_EPS})")
fig.colorbar(im0, ax=axs[0], label="temp", shrink=0.8)

im1 = axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title(f"init - target (c_k={INIT_C_K}, c_eps={INIT_C_EPS})")

im2 = axs[2].pcolormesh(xt, yt, bias_best.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title(f"best (iter {best_iter}) - target\n(c_k={best_ck:.4f}, c_eps={best_ceps:.4f})")

im3 = axs[3].pcolormesh(xt, yt, bias_final.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[3].set_title(f"final - target\n(c_k={FINAL_C_K:.4f}, c_eps={FINAL_C_EPS:.4f})")
fig.colorbar(im3, ax=[axs[1], axs[2], axs[3]], label="bias", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"Report 8: temperature bias after {N_STEPS} steps, surface level (global_4deg_learning + jcm)")

snap_path = os.path.join(OUT_DIR, "temp_snapshot.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"saved {snap_path}")
