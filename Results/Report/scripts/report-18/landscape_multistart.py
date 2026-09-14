"""Report 18 addendum: loss-landscape grid scan + 3-start multistart for the
stratification loss.

Direct counterpart to report-8's `landscape_multistart.py`, on the *same* grid
ranges and the same three starting points, so the two landscapes can be laid side
by side: report-8 scanned a full-column temperature loss and found a wide, flat,
uninformative band in `c_eps`. This scans the report-17 stratification metric,
mean(|dT/dz|) over the top 3 interfaces, which forward-mode predicted to be 62x
better conditioned along that same weak direction.

Grid is forward-only (no gradient); the three optimizations use the report-15
recipe (adam + cosine-decay lr, N_ITERS=30). Both checkpoint to .npy and resume.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    LEARNING_RATE, N_INTERFACES, N_ITERS, N_STEPS, REPO_ROOT,
    TRUE_C_EPS, TRUE_C_K, acquire_lock, best_iterate, build, load_trajectory,
    make_kernels, release_lock,
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

# Same ranges and resolution as report-8's scan, so the two landscapes are
# directly comparable.
CK_RANGE = (0.02, 0.18)
CEPS_RANGE = (0.25, 0.95)
GRID_N = int(os.environ.get("REPORT18_GRID_N", 12))

STARTS = [
    (0.05, 0.40),
    (0.15, 0.85),
    (0.03, 0.55),
]

LOCK = acquire_lock(OUT_DIR)

cpl, initial_state, grid = build()
strat, make_loss = make_kernels(cpl, initial_state, grid)
COLUMN_MASK = grid["column_mask"]

strat_jit = jax.jit(strat)
print(f"columns in loss: {int(COLUMN_MASK.sum())} / {COLUMN_MASK.size}", flush=True)

print(f"\nGenerating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}, N_STEPS={N_STEPS}) ...", flush=True)
target = jax.lax.stop_gradient(strat_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
loss_fn = make_loss(target)
loss_jit = jax.jit(loss_fn)
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# --- grid scan ---------------------------------------------------------------

GRID_PATH = os.path.join(OUT_DIR, "loss_grid.npy")
CK_PATH = os.path.join(OUT_DIR, "ck_grid.npy")
CEPS_PATH = os.path.join(OUT_DIR, "ceps_grid.npy")

ck_grid = np.linspace(*CK_RANGE, GRID_N)
ceps_grid = np.linspace(*CEPS_RANGE, GRID_N)

if os.path.exists(GRID_PATH) and os.path.exists(CK_PATH) and np.array_equal(np.load(CK_PATH), ck_grid):
    loss_grid = np.load(GRID_PATH)
    print(f"resuming grid scan ({int(np.isfinite(loss_grid).sum())} / {GRID_N ** 2} points done)", flush=True)
else:
    loss_grid = np.full((GRID_N, GRID_N), np.nan)

print(f"\nGrid scan: {GRID_N}x{GRID_N} = {GRID_N ** 2} forward evals ...", flush=True)
t_grid = time.time()
for i, ck in enumerate(ck_grid):
    for j, ce in enumerate(ceps_grid):
        if np.isfinite(loss_grid[i, j]):
            continue
        t0 = time.time()
        loss_grid[i, j] = float(loss_jit({"c_k": jnp.asarray(ck), "c_eps": jnp.asarray(ce)}))
        print(f"  ({i:2d},{j:2d}) c_k={ck:.4f} c_eps={ce:.4f}  loss={loss_grid[i, j]:.6e}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        np.save(GRID_PATH, loss_grid)
        np.save(CK_PATH, ck_grid)
        np.save(CEPS_PATH, ceps_grid)
print(f"grid scan wall time: {time.time() - t_grid:.1f}s", flush=True)

# --- multistart optimizations ------------------------------------------------

def optimize(idx, start_ck, start_ceps):
    path = os.path.join(OUT_DIR, f"start_{idx}_trajectory.npy")
    traj, complete = load_trajectory(path, N_ITERS, (start_ck, start_ceps))
    if complete:
        return traj

    params = {"c_k": jnp.asarray(traj[-1][0]), "c_eps": jnp.asarray(traj[-1][1])}
    schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    for iteration in range(len(traj) - 1, N_ITERS):
        t0 = time.time()
        loss_value, grads = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        traj.append((float(params["c_k"]), float(params["c_eps"]), float(loss_value)))
        print(f"  start {idx} iter {iteration:2d}  loss={float(loss_value):.6e}  "
              f"c_k={float(params['c_k']):.4f}  c_eps={float(params['c_eps']):.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        np.save(path, np.array([[t[0], t[1], t[2] if t[2] is not None else np.nan] for t in traj]))
    return traj


trajectories = []
for idx, (ck0, ce0) in enumerate(STARTS):
    print(f"\n=== start {idx}: c_k={ck0}, c_eps={ce0} ===", flush=True)
    trajectories.append(optimize(idx, ck0, ce0))

print(f"\n{'start':>16s} {'best iter':>9s} {'best (c_k, c_eps, loss)':>34s} "
      f"{'end (c_k, c_eps, loss)':>34s} {'c_eps drift':>12s}", flush=True)
summary_rows = []
for (ck0, ce0), traj in zip(STARTS, trajectories):
    bi, bl, bck, bce = best_iterate(traj)
    eck, ece, el = traj[-1]
    drift = ece - ce0
    summary_rows.append((ck0, ce0, bi, bck, bce, bl, eck, ece, el, drift))
    print(f"  ({ck0:.2f}, {ce0:.2f}) {bi:9d} "
          f"  ({bck:.4f}, {bce:.4f}, {bl:.4e}) "
          f"  ({eck:.4f}, {ece:.4f}, {el:.4e}) "
          f"  {drift:+.3f} ({100 * abs(drift) / ce0:.0f}%)", flush=True)

# --- figure ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))
CK, CEPS = np.meshgrid(ck_grid, ceps_grid, indexing="ij")
with np.errstate(invalid="ignore"):
    cf = ax.contourf(CK, CEPS, np.log10(loss_grid), levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="log10(loss)")
ax.contour(CK, CEPS, np.log10(loss_grid), levels=12, colors="white", linewidths=0.4, alpha=0.5)

colors = ["tab:red", "tab:orange", "magenta"]
for (ck0, ce0), traj, c in zip(STARTS, trajectories, colors):
    cks = [t[0] for t in traj]
    ceps = [t[1] for t in traj]
    ax.plot(cks, ceps, "-o", color=c, ms=3.5, lw=1.4, label=f"start ({ck0}, {ce0})")
    ax.plot(cks[0], ceps[0], "o", color=c, ms=10, markeredgecolor="black")
    ax.plot(cks[-1], ceps[-1], "s", color=c, ms=9, markeredgecolor="black")
ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="white", ms=24, markeredgecolor="black", zorder=6, label="true")

ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.set_title(f"Report 18: loss landscape + GD trajectories, N_STEPS={N_STEPS}\n"
             f"loss = mean(|dT/dz|) over the top {N_INTERFACES} interfaces")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
p = os.path.join(OUT_DIR, "landscape_multistart.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"\nsaved {p}", flush=True)

np.save(os.path.join(OUT_DIR, "multistart_summary.npy"),
        np.array([summary_rows], dtype=object), allow_pickle=True)
release_lock(LOCK)
print("done.", flush=True)
