"""Report 20: report-15's exact fit, at N_STEPS=10 instead of 20.

Report-19 found the chaotic decorrelation floor switches on between day 10 and
day 15: at N=10 a 30% c_eps error is ~1e9 times louder than a 1e-5 perturbation,
against 2-56 at N=20. If that floor is what actually limited reports 8/15/16/18,
then the *simplest* observable -- the surface temperature loss that stalled at
c_eps = 0.489 in report-15 -- should recover c_eps once the rollout is short
enough that no floor exists.

Everything except the rollout length is report-15's recipe:
  loss   = sum((temp_surface - target)^2) over ocean cells, masked by maskT
  optim  = optax.adam + cosine decay to 0, peak lr 2e-2, N_ITERS=30
  start  = c_k 0.05, c_eps 0.40      true = c_k 0.1, c_eps 0.7

so the comparison against report-15 isolates N_STEPS.
"""

import os
import sys
import time

# common.py reads these at import time; N_STEPS is the whole point of this run.
os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "30")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import (  # noqa: E402
    INIT_C_EPS, INIT_C_K, LEARNING_RATE, N_ITERS, N_STEPS, REPO_ROOT,
    TRUE_C_EPS, TRUE_C_K, acquire_lock, best_iterate, build, load_trajectory, release_lock,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20"))
os.makedirs(OUT_DIR, exist_ok=True)
LOCK = acquire_lock(OUT_DIR)

cpl, initial_state, grid = build()
MASK_NP = grid["maskT"]
SURF = MASK_NP.shape[2] - 1
SURFACE_MASK = jnp.asarray(MASK_NP[:, :, SURF])
XT, YT = grid["longitude"], grid["latitude"]
print(f"N_STEPS={N_STEPS}, N_ITERS={N_ITERS}; surface ocean cells in loss: "
      f"{int(MASK_NP[:, :, SURF].sum())} / {MASK_NP[:, :, SURF].size}", flush=True)


def surface_temp(c_k, c_eps):
    cs = initial_state._component_state("OCN")
    payload = set_veros_variable(cs.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", cs.with_payload(payload))
    p = cpl.run(state, output=None)._component_state("OCN").payload
    return p.variables.temp[2:-2, 2:-2, SURF, p.variables.tau]


temp_jit = jax.jit(surface_temp)
print(f"\ngenerating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...", flush=True)
t0 = time.time()
target = jax.lax.stop_gradient(temp_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
print(f"  done ({time.time() - t0:.1f}s incl. compile)", flush=True)


def loss_fn(params):
    t = surface_temp(params["c_k"], params["c_eps"])
    return jnp.sum(jnp.where(SURFACE_MASK, t - target, 0.0) ** 2)


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

TRAJ = os.path.join(OUT_DIR, "trajectory.npy")
trajectory, _ = load_trajectory(TRAJ, N_ITERS, (INIT_C_K, INIT_C_EPS))
params = {"c_k": jnp.asarray(trajectory[-1][0]), "c_eps": jnp.asarray(trajectory[-1][1])}
schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)

# Optional gradient clipping. The unclipped N=10/100-iteration run hit a single
# outlier at iteration 24 -- dL/dc_k = -1.17e3 and dL/dc_eps = +2.57e2, some 200x
# and 400x their neighbours, with flipped signs -- at the best point any run in
# this series has reached (c_k = 0.1000, c_eps = 0.5444, still climbing). The
# gradients were normal again by the next iteration, but adam's first moment
# carried the spike for ~25 iterations and walked the run out to c_k = 0.171,
# c_eps = 0.472. Typical global norms here are 5-40, so a clip at 50 passes every
# healthy gradient and caps only the outliers.
# Default off, so the 30-iteration run stays exactly report-15's recipe.
CLIP = float(os.environ.get("REPORT20_CLIP", "0") or 0)
if CLIP > 0:
    print(f"gradient clipping ON: global norm capped at {CLIP}", flush=True)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(schedule))
else:
    optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)
best_iter, best_loss, best_ck, best_ceps = best_iterate(trajectory)

print(f"\n{'iter':>4s} {'loss':>13s} {'c_k':>9s} {'c_eps':>9s} {'dL/dc_k':>12s} {'dL/dc_eps':>12s} {'s':>6s}", flush=True)
for it in range(len(trajectory) - 1, N_ITERS):
    t0 = time.time()
    prev = (float(params["c_k"]), float(params["c_eps"]))
    lv, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    trajectory.append((float(params["c_k"]), float(params["c_eps"]), float(lv)))
    print(f"{it:4d} {float(lv):13.5e} {float(params['c_k']):9.4f} {float(params['c_eps']):9.4f} "
          f"{float(grads['c_k']):12.3e} {float(grads['c_eps']):12.3e} {time.time() - t0:6.0f}", flush=True)
    np.save(TRAJ, np.array([[a, b, c if c is not None else np.nan] for a, b, c in trajectory]))
    if float(lv) < best_loss:
        best_loss, best_iter = float(lv), it
        best_ck, best_ceps = prev

FC_K, FC_EPS, FLOSS = trajectory[-1]
print(f"\nfinal   c_k={FC_K:.4f} ({100 * abs(FC_K - TRUE_C_K) / TRUE_C_K:.1f}% err)  "
      f"c_eps={FC_EPS:.4f} ({100 * abs(FC_EPS - TRUE_C_EPS) / TRUE_C_EPS:.1f}% err)", flush=True)
print(f"best    c_k={best_ck:.4f}  c_eps={best_ceps:.4f}  loss={best_loss:.5e} (iter {best_iter})", flush=True)
losses = np.array([t[2] for t in trajectory[1:]], float)
print(f"loss range {np.nanmin(losses):.4e} to {np.nanmax(losses):.4e} "
      f"({np.nanmax(losses) / np.nanmin(losses):.1f}x)", flush=True)
print(f"\nreport-15 at N_STEPS=20 for comparison: c_k=0.0876 (12% err), c_eps=0.4892 (30% err)", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(range(len(losses)), losses, "o-", color="darkgreen")
axs[0].set_yscale("log")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (masked surface temp SSE)")
axs[0].set_title("Loss vs iteration")
axs[0].grid(alpha=0.3)
cks = [t[0] for t in trajectory]
ces = [t[1] for t in trajectory]
axs[1].plot(cks, ces, "-o", color="darkgreen", ms=4, label="N=10 (this run)")
axs[1].plot(cks[0], ces[0], "o", color="black", ms=9, label="start")
axs[1].plot(cks[-1], ces[-1], "s", color="darkgreen", ms=10, label="final")
axs[1].plot(0.0876, 0.4892, "X", color="tab:red", ms=12, label="report-15 final (N=20)")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=22, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)
fig.suptitle(f"Report 20: surface-temperature fit at N_STEPS={N_STEPS} (report-15's recipe, shorter rollout)")
fig.tight_layout()
p = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}", flush=True)

release_lock(LOCK)
print("done.", flush=True)
