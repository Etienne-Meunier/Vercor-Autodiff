"""Report 21: report-20's N=10 fit under three alternative observables.

    REPORT21_OBS=mld       instantaneous MLD of the day-10 state
    REPORT21_OBS=temp_avg  surface temperature averaged over days 8, 9, 10
    REPORT21_OBS=mld_avg   MLD averaged over days 8, 9, 10
    REPORT21_OBS=temp      instantaneous surface temperature (= report-20-clip;
                           here only to re-derive the baseline if needed)

Everything except the observable is report-20's: N_STEPS=10, N_ITERS=100,
optax.adam on a cosine schedule from lr 2e-2 to 0, start c_k=0.05 / c_eps=0.40,
true c_k=0.1 / c_eps=0.7, gradient global norm clipped (see common21.CLIP_FACTOR).

The question is whether averaging the observable over the last three days damps
the gradient noise report-20 saw -- a 3-4x noisy loss curve and, unclipped, a
single ~200x gradient outlier that adam's momentum carried for ~25 iterations.
So the per-iteration gradient is checkpointed alongside the parameters and the
loss, and the run prints the gradient norm and its ratio to the running median
every iteration; those are the numbers the comparison is made on.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

import common21  # noqa: E402
from common21 import (  # noqa: E402
    AVG_WINDOW, CLIP_FACTOR, INIT_C_EPS, INIT_C_K, LEARNING_RATE, N_ITERS, N_STEPS,
    REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build, make_observable,
)
from common import acquire_lock, best_iterate, release_lock  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

OBS = os.environ.get("REPORT21_OBS", "mld")
if OBS not in ("temp", "temp_avg", "mld", "mld_avg", "tsdiff_avg"):
    raise SystemExit(f"REPORT21_OBS must be one of temp/temp_avg/mld/mld_avg/tsdiff_avg, got {OBS!r}")
AVERAGED = OBS.endswith("_avg")

OUT_DIR = os.environ.get(
    "REPORT21_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", f"report-21-{OBS}")
)
os.makedirs(OUT_DIR, exist_ok=True)
LOCK = acquire_lock(OUT_DIR)

UNITS = {"temp": "K^2", "temp_avg": "K^2", "mld": "m^2", "mld_avg": "m^2",
         "tsdiff_avg": "variance-normalised"}[OBS]
print(f"observable={OBS} ({'mean over the last %d states' % AVG_WINDOW if AVERAGED else 'final state only'}), "
      f"N_STEPS={N_STEPS}, N_ITERS={N_ITERS}, loss in {UNITS}", flush=True)

couplers, initial_state, grid = build(segmented=AVERAGED)
observable = make_observable(OBS, couplers, initial_state, grid)
obs_jit = jax.jit(observable)

print(f"\ngenerating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...", flush=True)
t0 = time.time()
target, target_valid = obs_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target = jax.lax.stop_gradient(target)
TARGET_VALID = jnp.asarray(np.asarray(target_valid))  # frozen: the loss domain must
                                                      # not move between iterations
print(f"  done ({time.time() - t0:.1f}s incl. compile)", flush=True)

_valid_np = np.asarray(TARGET_VALID)
_ocean_np = grid["maskT"][:, :, grid["surf"]]
print(f"cells in loss: {int(_valid_np.sum())} / {int(_ocean_np.sum())} surface ocean cells "
      f"({_valid_np.size} total surface cells)", flush=True)
_tgt = np.asarray(target)[_valid_np]
print(f"target range: {_tgt.min():.4f} to {_tgt.max():.4f} (mean {_tgt.mean():.4f})", flush=True)
W = common21.loss_weights(OBS, target, TARGET_VALID)
print(f"loss weights: {np.asarray(W)}", flush=True)


def loss_fn(params):
    """Sum of squared error over the frozen loss domain.

    Columns that lose their MLD in a candidate run are dropped too, so the mask is
    the intersection with the target's -- a column with no MLD contributes 0 rather
    than the difference against a placeholder.
    """
    field, valid = observable(params["c_k"], params["c_eps"])
    diff = jnp.where(TARGET_VALID & valid, field - target, 0.0)
    return jnp.sum(W * diff ** 2)


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

TRAJ = os.path.join(OUT_DIR, "trajectory.npy")


def load_trajectory(path, n_iters, start):
    """Load a checkpointed trajectory, but ONLY if it is complete.

    Report-18's policy, widened to this script's 5-column rows
    (c_k, c_eps, loss, dL/dc_k, dL/dc_eps). Resuming a partial run is
    scientifically wrong here: `optimizer.init` rebuilds adam's moment estimates
    from zero, so a resumed run is not the same recipe as an uninterrupted one and
    is not comparable to report-20. It would also re-measure the clip threshold at
    the resume point rather than at the start point. So a COMPLETE trajectory is
    reused and a PARTIAL one is discarded.
    """
    if not os.path.exists(path):
        return [(start[0], start[1], np.nan, np.nan, np.nan)]
    traj = [tuple(r) for r in np.load(path).tolist()]
    if len(traj) - 1 >= n_iters:
        print(f"  reusing COMPLETE trajectory from {os.path.basename(path)} "
              f"({len(traj) - 1} iterations)", flush=True)
        return traj
    print(f"  discarding PARTIAL trajectory in {os.path.basename(path)} "
          f"({len(traj) - 1}/{n_iters} iters): resuming would reset adam's moment "
          f"estimates and break comparability. Restarting from scratch.", flush=True)
    return [(start[0], start[1], np.nan, np.nan, np.nan)]


def save(traj):
    np.save(TRAJ, np.array([[c if c is not None else np.nan for c in row] for row in traj], float))


trajectory = load_trajectory(TRAJ, N_ITERS, (INIT_C_K, INIT_C_EPS))
params = {"c_k": jnp.asarray(trajectory[-1][0]), "c_eps": jnp.asarray(trajectory[-1][1])}
schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)

header = (f"\n{'iter':>4s} {'loss':>13s} {'c_k':>9s} {'c_eps':>9s} {'dL/dc_k':>12s} "
          f"{'dL/dc_eps':>12s} {'|g|':>11s} {'|g|/med':>8s} {'s':>6s}")

optimizer = None
opt_state = None
CLIP = None
grad_norms = []

if len(trajectory) - 1 < N_ITERS:
    # Measure the gradient at the start point before building the optimizer: the
    # clip threshold is CLIP_FACTOR times this norm (see common21.CLIP_FACTOR),
    # which reproduces report-20's fixed 50 for report-20's observable while
    # staying meaningful for a loss carried in different units.
    print("\nmeasuring the gradient at the start point to set the clip threshold ...", flush=True)
    t0 = time.time()
    loss0, grad0 = value_and_grad_fn(params)
    norm0 = float(np.sqrt(float(grad0["c_k"]) ** 2 + float(grad0["c_eps"]) ** 2))
    CLIP = CLIP_FACTOR * norm0
    print(f"  loss {float(loss0):.5e}, dL/dc_k {float(grad0['c_k']):.4e}, "
          f"dL/dc_eps {float(grad0['c_eps']):.4e}, |g| {norm0:.4e} ({time.time() - t0:.1f}s incl. compile)", flush=True)
    print(f"  gradient clipping ON: global norm capped at {CLIP:.4e} "
          f"({CLIP_FACTOR} x the start-point norm)", flush=True)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(schedule))
    opt_state = optimizer.init(params)
    print(header, flush=True)

best_iter, best_loss, best_ck, best_ceps = best_iterate(trajectory)
for it in range(len(trajectory) - 1, N_ITERS):
    t0 = time.time()
    prev = (float(params["c_k"]), float(params["c_eps"]))
    if it == 0:
        lv, grads = loss0, grad0  # already paid for above
    else:
        lv, grads = value_and_grad_fn(params)
    gk, ge = float(grads["c_k"]), float(grads["c_eps"])
    gn = float(np.sqrt(gk ** 2 + ge ** 2))
    grad_norms.append(gn)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    trajectory.append((float(params["c_k"]), float(params["c_eps"]), float(lv), gk, ge))
    med = float(np.median(grad_norms))
    print(f"{it:4d} {float(lv):13.5e} {float(params['c_k']):9.4f} {float(params['c_eps']):9.4f} "
          f"{gk:12.3e} {ge:12.3e} {gn:11.4e} {gn / med:8.2f} {time.time() - t0:6.0f}", flush=True)
    save(trajectory)
    if float(lv) < best_loss:
        best_loss, best_iter = float(lv), it
        best_ck, best_ceps = prev

FC_K, FC_EPS, FLOSS = trajectory[-1][0], trajectory[-1][1], trajectory[-1][2]
print(f"\nfinal   c_k={FC_K:.4f} ({100 * abs(FC_K - TRUE_C_K) / TRUE_C_K:.1f}% err)  "
      f"c_eps={FC_EPS:.4f} ({100 * abs(FC_EPS - TRUE_C_EPS) / TRUE_C_EPS:.1f}% err)", flush=True)
print(f"best    c_k={best_ck:.4f} ({100 * abs(best_ck - TRUE_C_K) / TRUE_C_K:.1f}% err)  "
      f"c_eps={best_ceps:.4f} ({100 * abs(best_ceps - TRUE_C_EPS) / TRUE_C_EPS:.1f}% err)  "
      f"loss={best_loss:.5e} (iter {best_iter})", flush=True)

losses = np.array([t[2] for t in trajectory[1:]], float)
print(f"loss range {np.nanmin(losses):.4e} to {np.nanmax(losses):.4e} "
      f"({np.nanmax(losses) / np.nanmin(losses):.1f}x)", flush=True)

# Gradient-stability summary: the point of the report. A time average that damps
# the gradient noise should show a smaller max/median norm ratio and a smaller
# step-to-step direction change than report-20's instantaneous surface temperature.
gvec = np.array([[t[3], t[4]] for t in trajectory[1:]], float)
gnorm = np.linalg.norm(gvec, axis=1)
unit = gvec / np.where(gnorm[:, None] == 0, 1.0, gnorm[:, None])
cos = np.sum(unit[1:] * unit[:-1], axis=1)
print(f"\ngradient norm: median {np.median(gnorm):.4e}, max {gnorm.max():.4e} "
      f"(max/median {gnorm.max() / np.median(gnorm):.1f}x), "
      f"{int((gnorm > 10 * np.median(gnorm)).sum())} iterations above 10x the median", flush=True)
print(f"cos angle between consecutive gradients: median {np.median(cos):.3f}, "
      f"{int((cos < 0).sum())} / {cos.size} sign reversals", flush=True)
print(f"clipped iterations: {int((gnorm > CLIP).sum()) if CLIP else 'n/a (nothing rerun)'}"
      f"{'' if CLIP is None else ' / %d' % gnorm.size}", flush=True)

fig, axs = plt.subplots(1, 3, figsize=(17, 5))
axs[0].plot(range(len(losses)), losses, "o-", color="darkgreen", ms=3)
axs[0].set_yscale("log")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel(f"loss (SSE, {UNITS})")
axs[0].set_title(f"Loss vs iteration ({OBS})")
axs[0].grid(alpha=0.3)

axs[1].plot(range(gnorm.size), gnorm, "o-", color="tab:purple", ms=3)
axs[1].axhline(np.median(gnorm), color="black", ls="--", lw=1, label="median")
if CLIP:
    axs[1].axhline(CLIP, color="tab:red", ls=":", lw=1.5, label=f"clip ({CLIP:.2e})")
axs[1].set_yscale("log")
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("|dL/d(c_k, c_eps)|")
axs[1].set_title("Gradient norm")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ces = [t[1] for t in trajectory]
axs[2].plot(cks, ces, "-o", color="darkgreen", ms=3, label=f"{OBS} (this run)")
axs[2].plot(cks[0], ces[0], "o", color="black", ms=9, label="start")
axs[2].plot(cks[-1], ces[-1], "s", color="darkgreen", ms=10, label="final")
axs[2].plot(0.0912, 0.5624, "X", color="tab:red", ms=11, label="report-20 final (inst. temp)")
axs[2].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=22, markeredgecolor="black", label="true", zorder=5)
axs[2].set_xlabel("c_k")
axs[2].set_ylabel("c_eps")
axs[2].set_title("Parameter trajectory")
axs[2].legend(fontsize=8)
axs[2].grid(alpha=0.3)

fig.suptitle(f"Report 21: N_STEPS={N_STEPS} fit, observable = {OBS}"
             + (f" (mean over the last {AVG_WINDOW} days)" if AVERAGED else ""))
fig.tight_layout()
p = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"saved {p}", flush=True)

release_lock(LOCK)
print("done.", flush=True)
