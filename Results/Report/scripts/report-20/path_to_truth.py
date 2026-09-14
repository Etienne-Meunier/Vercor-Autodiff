"""Report 20: is the fit stuck in a local minimum, or just badly conditioned?

The loss at the true parameters is exactly zero by construction (the target is
generated from them), yet the fit settles at 4.6e-2. Two explanations:

  (a) a local minimum -- a barrier separates the fitted point from the truth, so
      no descent method starting there can reach it;
  (b) no barrier -- the loss descends all the way, but along a direction the
      optimizer is not taking (report-18's diagonal c_k/c_eps valley), so the
      run slides along the valley instead of down it.

These are distinguishable by walking a straight line from the fitted point to the
truth and evaluating the loss. Monotone descent to zero rules out (a).

Also evaluates the loss at the truth itself and at the two endpoints' own
gradients, and scans perpendicular to the path for completeness.
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "100")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, best_iterate, build  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20-clip"))
traj = [tuple(r) for r in np.load(os.path.join(OUT_DIR, "trajectory.npy")).tolist()]
best_iter, best_loss, FIT_C_K, FIT_C_EPS = best_iterate(traj)
print(f"fitted (best loss, iter {best_iter}): c_k={FIT_C_K:.4f}, c_eps={FIT_C_EPS:.4f}, "
      f"loss={best_loss:.5e}", flush=True)

cpl, initial_state, grid = build()
MASK = jnp.asarray(grid["maskT"][:, :, grid["maskT"].shape[2] - 1])
SURF = grid["maskT"].shape[2] - 1


def surface_temp(c_k, c_eps):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", c_k)
    p = set_veros_variable(p, "c_eps", c_eps)
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    pay = cpl.run(st, output=None)._component_state("OCN").payload
    return pay.variables.temp[2:-2, 2:-2, SURF, pay.variables.tau]


target = jax.lax.stop_gradient(jax.jit(surface_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))


def loss_fn(params):
    t = surface_temp(params["c_k"], params["c_eps"])
    return jnp.sum(jnp.where(MASK, t - target, 0.0) ** 2)


vg = jax.jit(jax.value_and_grad(loss_fn))

print(f"\nstraight line from the fitted point to the truth ({N_STEPS}-day rollout)", flush=True)
print(f"{'t':>5s} {'c_k':>8s} {'c_eps':>8s} {'loss':>13s} {'dL/dc_k':>12s} {'dL/dc_eps':>12s}", flush=True)
rows = []
for t in np.linspace(0.0, 1.0, 11):
    a = FIT_C_K + t * (TRUE_C_K - FIT_C_K)
    b = FIT_C_EPS + t * (TRUE_C_EPS - FIT_C_EPS)
    t0 = time.time()
    lv, g = vg({"c_k": jnp.asarray(a), "c_eps": jnp.asarray(b)})
    rows.append((t, a, b, float(lv), float(g["c_k"]), float(g["c_eps"])))
    print(f"{t:5.2f} {a:8.4f} {b:8.4f} {float(lv):13.5e} {float(g['c_k']):12.4e} "
          f"{float(g['c_eps']):12.4e}   [{time.time() - t0:.0f}s]", flush=True)

np.save(os.path.join(OUT_DIR, "path_to_truth.npy"), np.array(rows))
L = np.array([r[3] for r in rows])
monotone = bool(np.all(np.diff(L) <= 0))
print(f"\nmonotone decreasing along the path: {monotone}", flush=True)
print(f"  loss at fitted point {L[0]:.5e} -> at truth {L[-1]:.5e}", flush=True)
if not monotone:
    i = int(np.argmax(L))
    print(f"  highest point on the path: t={rows[i][0]:.2f}, loss={L[i]:.5e} "
          f"({L[i] / L[0]:.2f}x the fitted point)", flush=True)
print("done.", flush=True)
