"""Report 10: repeat report-8's calibration (n=20, direct value_and_grad,
adam + cosine-decay lr, same true/init c_k/c_eps), but starting from an
atmosphere that has already been spun up for SPINUP_DAYS, instead of the
cold day-0 start every earlier report used.

Motivation (report-9): JCM's domain-mean forcing statistics settle into a
stationary regime after an initial ~50-150 day transient, though the
spatial pattern itself never stops producing new chaotic structure. This
tests whether starting the n=20 gradient window *after* that initial
transient measurably changes gradient trustworthiness (FD-vs-AD
agreement) or fit behavior, compared to report-3's original cold-start
n=20 check (7.3% rel err) and report-8's cold-start fit.

Two parts:
1. Direct FD-vs-AD spot check at n=20 from the spun-up state (same
   methodology as report-3's bracket), for direct numeric comparison.
2. A full report-8-style fit (target-gen + adam/cosine-decay descent) from
   the spun-up state.

Forward-only spin-up (no grad, cheap); the calibration window itself is
gradient-based as in report-8.
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

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-10")
os.makedirs(OUT_DIR, exist_ok=True)

START = datetime(2001, 1, 3, 0, 0, 0)
SPINUP_DAYS = 50
N_STEPS = 20  # same as report-8
N_ITERS = 30
LEARNING_RATE = 2e-2
PATIENCE = 6
PATIENCE_REL_TOL = 5e-3
FD_EPS = 1e-2

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4
Z_LEVEL = -1

DTYPE = DTypePolicy(enable_x64=True)

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
    config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)),
)
lnd = jcm_setup.land
atm = jcm_setup.atmosphere
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def make_coupler(start_date, n_steps):
    clock = Clock(start=start_date, dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    return Coupler(
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=exchanges,
        run_order=["OCN", "LND", "ATM"],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTYPE),
    )


print(f"Spinning up {SPINUP_DAYS} days forward-only (no grad) from {START} ...", flush=True)
spinup_cpl = make_coupler(START, SPINUP_DAYS)
t0 = time.time()
spun_up_state = jax.jit(lambda s: spinup_cpl.run(s, output=None))(spinup_cpl.initial_state())
spun_up_state = jax.lax.stop_gradient(spun_up_state)
print(f"spin-up done ({time.time() - t0:.1f}s)", flush=True)

calib_start = START + timedelta(days=SPINUP_DAYS)
calib_cpl = make_coupler(calib_start, N_STEPS)
calib_cpl.initial_state()  # eager prep before jit trace, per report-6/7 lesson


def final_ocn_payload(c_k, c_eps):
    component_state = spun_up_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = spun_up_state._with_component_state("OCN", component_state.with_payload(payload))
    result = calib_cpl.run(state, output=None)
    return result._component_state("OCN").payload


def final_ocn_temp(c_k, c_eps):
    return final_ocn_payload(c_k, c_eps).variables.temp


# --- Part 1: direct FD-vs-AD spot check at n=20, from the spun-up state ---
print(f"\nFD-vs-AD spot check at n={N_STEPS}, post-{SPINUP_DAYS}-day spinup (c_k={TRUE_C_K}) ...", flush=True)


def loss_scalar(c_k):
    return jnp.sum(final_ocn_temp(c_k, jnp.asarray(TRUE_C_EPS)) ** 2)


ad_value, ad_grad = jax.value_and_grad(loss_scalar)(jnp.asarray(TRUE_C_K))
loss_plus = float(loss_scalar(jnp.asarray(TRUE_C_K + FD_EPS)))
loss_minus = float(loss_scalar(jnp.asarray(TRUE_C_K - FD_EPS)))
fd_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
rel_err = abs(float(ad_grad) - fd_grad) / abs(fd_grad)
print(f"AD grad:  {float(ad_grad):.4e}")
print(f"FD grad:  {fd_grad:.4e}  (eps={FD_EPS})")
print(f"rel err:  {rel_err:.1%}  (report-3 cold-start n=20 reference: 7.3%)", flush=True)

# --- Part 2: report-8-style fit from the spun-up state ---
print(f"\nGenerating synthetic target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}, post-spinup) ...", flush=True)
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
no_improve_count = 0

print(f"\n{'iter':>4s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}  {'lr':>10s}  {'wall_s':>8s}")
print(f"{'init':>4s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")

t_start = time.time()
for iteration in range(N_ITERS):
    t0 = time.time()
    loss_value, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    dt = time.time() - t0

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    loss_val = float(loss_value)
    trajectory.append((ck_val, ceps_val, loss_val))
    print(f"{iteration:4d}  {loss_val:14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  {float(schedule(iteration)):10.4f}  {dt:8.1f}", flush=True)

    if loss_val < best_loss * (1 - PATIENCE_REL_TOL):
        best_loss, best_iter = loss_val, iteration
        best_ck, best_ceps = ck_val, ceps_val
        no_improve_count = 0
    else:
        no_improve_count += 1

    if no_improve_count >= PATIENCE:
        print(f"early stop: no >{PATIENCE_REL_TOL:.1%} loss improvement for {PATIENCE} iterations")
        break

total_wall = time.time() - t_start
FINAL_C_K, FINAL_C_EPS = trajectory[-1][0], trajectory[-1][1]
print(f"\ntotal descent wall time: {total_wall:.1f}s over {len(trajectory) - 1} iterations")
print("recovered c_k (final):  ", FINAL_C_K, " (true:", TRUE_C_K, ")")
print("recovered c_eps (final):", FINAL_C_EPS, " (true:", TRUE_C_EPS, ")")
print(f"best iteration: {best_iter}, loss={best_loss:.6e}, c_k={best_ck:.4f}, c_eps={best_ceps:.4f}")

# --- snapshots ---
print("\nrolling out snapshots ...", flush=True)
initial_payload = jax.jit(final_ocn_payload)(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS))
best_payload = jax.jit(final_ocn_payload)(jnp.asarray(best_ck), jnp.asarray(best_ceps))
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

bias_max = np.nanmax(np.abs(np.concatenate([
    bias_initial[~np.isnan(bias_initial)],
    bias_best[~np.isnan(bias_best)],
    bias_final[~np.isnan(bias_final)],
])))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
iters = list(range(len(trajectory) - 1))
losses = [t[2] for t in trajectory[1:]]
axs[0].plot(iters, losses, "o-", color="tab:cyan")
axs[0].axvline(best_iter, color="tab:orange", linestyle="--", alpha=0.7, label=f"best (iter {best_iter})")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared temp error)")
axs[0].set_yscale("log")
axs[0].set_title("Loss vs iteration (log scale)")
axs[0].legend()
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="tab:cyan", markersize=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(best_ck, best_ceps, "D", color="tab:orange", markersize=9, label="best", zorder=4)
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:cyan", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 10: c_k/c_eps recovery, {N_STEPS}-step rollout after {SPINUP_DAYS}-day atmosphere spin-up")
fig.tight_layout()
traj_path = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"saved {traj_path}")

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
fig.suptitle(f"Report 10: temperature bias after {N_STEPS} steps post-spinup, surface level")

snap_path = os.path.join(OUT_DIR, "temp_snapshot.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"saved {snap_path}")
