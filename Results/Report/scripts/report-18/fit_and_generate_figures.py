"""Report 18: c_k/c_eps fit at N_STEPS=20 against a stratification loss.

Identical recipe to report-15 (optax.adam + cosine-decay lr, N_ITERS=30,
N_STEPS=20, start c_k=0.05/c_eps=0.4) with one change: the observable is the
report-17 stratification metric, mean(|dT/dz|) over the top 3 interfaces, instead
of surface temperature.

Report-17 predicted this from forward-mode tangents alone, without running any
optimization: the metric carries 62x surface temperature's curvature along the
worst-determined parameter direction, stably across three evaluation points. This
script tests that prediction against an actual fit.

Script pair: `landscape_multistart.py` adds the grid scan and two more starts.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    INIT_C_EPS, INIT_C_K, LEARNING_RATE, N_INTERFACES, N_ITERS, N_STEPS,
    REPO_ROOT, TRUE_C_EPS, TRUE_C_K, acquire_lock, best_iterate, build,
    load_trajectory, make_kernels, release_lock,
)

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

OUT_DIR = os.environ.get(
    "REPORT18_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-18")
)
os.makedirs(OUT_DIR, exist_ok=True)

LOCK = acquire_lock(OUT_DIR)

cpl, initial_state, grid = build()
strat, make_loss = make_kernels(cpl, initial_state, grid)

COLUMN_MASK = grid["column_mask"]
XT, YT, ZT = grid["longitude"], grid["latitude"], grid["zt"]
N_COLS = int(COLUMN_MASK.sum())
print(f"interfaces averaged: top {N_INTERFACES}, between cell centres at "
      f"{np.round(ZT[-(N_INTERFACES + 1):], 1).tolist()} m", flush=True)
print(f"columns in loss: {N_COLS} / {COLUMN_MASK.size} surface cells", flush=True)

strat_jit = jax.jit(strat)

print(f"\nGenerating synthetic target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} "
      f"(N_STEPS={N_STEPS}) ...", flush=True)
t0 = time.time()
target = jax.lax.stop_gradient(strat_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
print(f"  done ({time.time() - t0:.1f}s, includes compile)", flush=True)
_t = np.asarray(target)[COLUMN_MASK]
print(f"target metric range: {_t.min():.4f} to {_t.max():.4f} deg C/m (mean {_t.mean():.4f})", flush=True)

loss_fn = make_loss(target)
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

TRAJ_PATH = os.path.join(OUT_DIR, "trajectory.npy")
trajectory, _complete = load_trajectory(TRAJ_PATH, N_ITERS, (INIT_C_K, INIT_C_EPS))

params = {"c_k": jnp.asarray(trajectory[-1][0]), "c_eps": jnp.asarray(trajectory[-1][1])}
schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)

best_iter, best_loss, best_ck, best_ceps = best_iterate(trajectory)
wall = []

print(f"\n{'iter':>4s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}  "
      f"{'dL/dc_k':>12s}  {'dL/dc_eps':>12s}  {'lr':>8s}  {'wall_s':>8s}", flush=True)
print(f"{'init':>4s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  "
      f"(true: {TRUE_C_K}, {TRUE_C_EPS})", flush=True)

t_start = time.time()
for iteration in range(len(trajectory) - 1, N_ITERS):
    t0 = time.time()
    prev_ck, prev_ceps = float(params["c_k"]), float(params["c_eps"])
    loss_value, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    dt = time.time() - t0
    wall.append(dt)

    ck_val, ceps_val, loss_val = float(params["c_k"]), float(params["c_eps"]), float(loss_value)
    trajectory.append((ck_val, ceps_val, loss_val))
    print(f"{iteration:4d}  {loss_val:14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  "
          f"{float(grads['c_k']):12.4e}  {float(grads['c_eps']):12.4e}  "
          f"{float(schedule(iteration)):8.4f}  {dt:8.1f}", flush=True)
    np.save(TRAJ_PATH, np.array([[t[0], t[1], t[2] if t[2] is not None else np.nan] for t in trajectory]))

    if loss_val < best_loss:
        best_loss, best_iter = loss_val, iteration
        best_ck, best_ceps = prev_ck, prev_ceps  # the params the loss was evaluated at

if wall:
    print(f"\ntotal descent wall time: {time.time() - t_start:.1f}s over {len(wall)} iterations "
          f"(mean {np.mean(wall):.1f}s/iter)", flush=True)

FINAL_C_K, FINAL_C_EPS, FINAL_LOSS = trajectory[-1]
print(f"recovered c_k (final):   {FINAL_C_K:.4f}  (true {TRUE_C_K}, "
      f"{100 * abs(FINAL_C_K - TRUE_C_K) / TRUE_C_K:.1f}% error)", flush=True)
print(f"recovered c_eps (final): {FINAL_C_EPS:.4f}  (true {TRUE_C_EPS}, "
      f"{100 * abs(FINAL_C_EPS - TRUE_C_EPS) / TRUE_C_EPS:.1f}% error)", flush=True)
print(f"best loss {best_loss:.6e} at c_k={best_ck:.4f}, c_eps={best_ceps:.4f} "
      f"(evaluated before iteration {best_iter}'s update)", flush=True)

losses = np.array([t[2] for t in trajectory[1:]], dtype=float)
print(f"loss range across the run: {np.nanmin(losses):.4e} to {np.nanmax(losses):.4e} "
      f"({np.nanmax(losses) / np.nanmin(losses):.1f}x)", flush=True)

# --- snapshots ---------------------------------------------------------------
print("\nrolling out init / best / final for the snapshot ...", flush=True)
init_s = np.asarray(strat_jit(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS)))
best_s = np.asarray(strat_jit(jnp.asarray(best_ck), jnp.asarray(best_ceps)))
final_s = np.asarray(strat_jit(jnp.asarray(FINAL_C_K), jnp.asarray(FINAL_C_EPS)))
target_np = np.asarray(target)


def masked(a):
    return np.where(COLUMN_MASK, a, np.nan)


target_field = masked(target_np)
biases = [masked(x) - target_field for x in (init_s, best_s, final_s)]
for label, b in zip(("init", "best", "final"), biases):
    rmse = float(np.sqrt(np.nanmean(b[~np.isnan(b)] ** 2)))
    print(f"  {label:>5s}: metric RMSE {rmse:.4e} deg C/m "
          f"({100 * rmse / float(np.nanmean(np.abs(target_field))):.2f}% of the mean metric)", flush=True)

np.save(os.path.join(OUT_DIR, "target_strat.npy"), target_field)
for label, b in zip(("initial", "best", "final"), biases):
    np.save(os.path.join(OUT_DIR, f"bias_{label}.npy"), b)

# --- figure 1: trajectory ----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
iters = list(range(len(trajectory) - 1))
axs[0].plot(iters, losses, "o-", color="teal")
axs[0].axvline(best_iter, color="tab:orange", ls="--", alpha=0.7, label=f"best (iter {best_iter})")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared metric error)")
axs[0].set_yscale("log")
axs[0].set_title("Loss vs iteration (log scale)")
axs[0].legend()
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="teal", ms=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", ms=8, label="start")
axs[1].plot(best_ck, best_ceps, "D", color="tab:orange", ms=9, label="best", zorder=4)
axs[1].plot(cks[-1], ceps[-1], "s", color="teal", ms=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", ms=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 18: c_k/c_eps recovery, {N_STEPS}-step coupled rollout, "
             f"mean(|dT/dz|) top-{N_INTERFACES} loss, adam + cosine-decay lr")
fig.tight_layout()
p = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}", flush=True)

# --- figure 2: metric snapshot -----------------------------------------------
bmax = max(np.nanmax(np.abs(b)) for b in biases)
fig, axs = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
im0 = axs[0].pcolormesh(XT, YT, target_field.T, cmap="viridis", shading="auto")
axs[0].set_title(f"target metric (c_k={TRUE_C_K}, c_eps={TRUE_C_EPS})")
fig.colorbar(im0, ax=axs[0], label="mean |dT/dz| (deg C/m)", shrink=0.8)

titles = [
    f"init - target (c_k={INIT_C_K}, c_eps={INIT_C_EPS})",
    f"best - target\n(c_k={best_ck:.4f}, c_eps={best_ceps:.4f})",
    f"final - target\n(c_k={FINAL_C_K:.4f}, c_eps={FINAL_C_EPS:.4f})",
]
for ax, b, title in zip(axs[1:], biases, titles):
    im = ax.pcolormesh(XT, YT, b.T, cmap="RdBu_r", vmin=-bmax, vmax=bmax, shading="auto")
    ax.set_title(title)
fig.colorbar(im, ax=list(axs[1:]), label="metric bias (deg C/m)", shrink=0.8)
for ax in axs:
    ax.set_xlabel("longitude")
axs[0].set_ylabel("latitude")
fig.suptitle(f"Report 18: stratification-metric bias after {N_STEPS} steps")
p = os.path.join(OUT_DIR, "strat_snapshot.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}", flush=True)

release_lock(LOCK)
print("\ndone.", flush=True)
