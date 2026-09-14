"""Report 20: Gauss-Newton from several starting points, run to convergence.

Two questions:
  (1) Does GN keep converging if it is allowed to continue? The earlier run
      stopped at iteration 5 with "damping search exhausted" after only 6
      damping retries (lambda capped near 8). Here the search gets 12 retries,
      so lambda can reach ~1e6 and the step degenerates smoothly toward a tiny
      gradient step rather than giving up.
  (2) How does the result depend on where you start? Seven starts at increasing
      distance from the truth, including two on the far side of it.

All starts share one build and one set of compiled functions, so the compile is
paid once rather than seven times.

Convergence: stop when the step is negligible, when the relative loss improvement
stays below tol for 3 consecutive iterations, or when damping is exhausted.
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "30")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR",
                         os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20-gn-multistart"))
os.makedirs(OUT_DIR, exist_ok=True)
MAX_ITERS = int(os.environ.get("REPORT20_MAX_ITERS", 25))
MAX_DAMPING = 12
TOL = 1e-4

# (label, c_k, c_eps) -- ordered by distance from the truth (0.1, 0.7)
STARTS = [
    ("very-near", 0.095, 0.670),
    ("near", 0.090, 0.620),
    ("mid", 0.080, 0.550),
    ("report-15/20 start", 0.050, 0.400),
    ("far", 0.030, 0.300),
    ("above", 0.150, 0.850),
    ("off-diagonal", 0.130, 0.500),
]

cpl, initial_state, grid = build()
SURF = grid["maskT"].shape[2] - 1
SMASK = jnp.asarray(grid["maskT"][:, :, SURF])


def surface_temp(a, b):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", a)
    p = set_veros_variable(p, "c_eps", b)
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    pay = cpl.run(st, output=None)._component_state("OCN").payload
    return pay.variables.temp[2:-2, 2:-2, SURF, pay.variables.tau]


print(f"generating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}, N_STEPS={N_STEPS}) ...", flush=True)
t0 = time.time()
TARGET = jax.lax.stop_gradient(jax.jit(surface_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
print(f"  done ({time.time() - t0:.0f}s)", flush=True)


def residual_pair(a, b):
    return jnp.where(SMASK, surface_temp(a, b) - TARGET, 0.0).ravel()


def _jac(a, b):
    # Two explicit JVPs. jax.jacfwd returns inf columns on this model (its vmap
    # over the tangent basis); separate jvp calls are correct and cost the same.
    _, t_k = jax.jvp(lambda x: residual_pair(x, b), (a,), (jnp.ones_like(a),))
    r, t_e = jax.jvp(lambda y: residual_pair(a, y), (b,), (jnp.ones_like(b),))
    return r, jnp.stack([t_k, t_e], axis=1)


jac_fn = jax.jit(_jac)
res_fn = jax.jit(residual_pair)


def perr(ck, ce):
    return float(np.hypot((ck - TRUE_C_K) / TRUE_C_K, (ce - TRUE_C_EPS) / TRUE_C_EPS) * 100)


summary = []
for idx, (label, ck0, ce0) in enumerate(STARTS):
    print(f"\n=== start {idx}: {label}  (c_k={ck0}, c_eps={ce0}, {perr(ck0, ce0):.1f}% from truth) ===", flush=True)
    theta = np.array([ck0, ce0], dtype=float)
    r = np.asarray(res_fn(jnp.asarray(theta[0]), jnp.asarray(theta[1])))
    loss = float(r @ r)
    traj = [(theta[0], theta[1], loss)]
    lam, stagnant, reason = 1e-3, 0, "iteration limit"
    print(f"  init   loss={loss:12.5e}  err={perr(*theta):6.1f}%", flush=True)

    for it in range(MAX_ITERS):
        t0 = time.time()
        _r, _J = jac_fn(jnp.asarray(theta[0]), jnp.asarray(theta[1]))
        J = np.asarray(_J)
        JTJ, JTr = J.T @ J, J.T @ r
        w = np.linalg.eigvalsh(JTJ)
        cond = float(w[-1] / w[0]) if w[0] > 0 else np.inf

        accepted = False
        for _ in range(MAX_DAMPING):
            scale = np.maximum(np.diag(JTJ), 1e-12 * max(float(np.max(np.abs(JTJ))), 1.0))
            A = JTJ + lam * np.diag(scale)
            try:
                step = -np.linalg.solve(A, JTr)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A, JTr, rcond=None)[0]
            cand = np.maximum(theta + step, 1e-4)
            r_new = np.asarray(res_fn(jnp.asarray(cand[0]), jnp.asarray(cand[1])))
            loss_new = float(r_new @ r_new)
            if loss_new < loss:
                rel = (loss - loss_new) / loss
                theta, r, loss = cand, r_new, loss_new
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                stagnant = stagnant + 1 if rel < TOL else 0
                break
            lam *= 5.0
        traj.append((theta[0], theta[1], loss))
        np.save(os.path.join(OUT_DIR, f"start_{idx}_trajectory.npy"), np.array(traj))
        print(f"  it {it:3d}  loss={loss:12.5e}  c_k={theta[0]:7.4f}  c_eps={theta[1]:7.4f}  "
              f"err={perr(*theta):6.1f}%  cond={cond:9.2e}  lam={lam:8.1e}"
              f"{'' if accepted else '  NO STEP'}  [{time.time() - t0:.0f}s]", flush=True)
        if not accepted:
            reason = "damping exhausted"
            break
        if np.linalg.norm(step) < 1e-9:
            reason = "step below tolerance"
            break
        if stagnant >= 3:
            reason = f"loss improvement < {TOL} for 3 iterations"
            break

    print(f"  stopped: {reason}", flush=True)
    summary.append((label, ck0, ce0, theta[0], theta[1], loss, len(traj) - 1, reason))

print(f"\n{'start':>20s} {'from':>16s} {'-> c_k':>9s} {'-> c_eps':>9s} {'err%':>7s} "
      f"{'loss':>12s} {'its':>4s}  reason", flush=True)
for label, a0, b0, a, b, l, n, why in summary:
    print(f"{label:>20s} {f'({a0},{b0})':>16s} {a:9.4f} {b:9.4f} {perr(a, b):7.1f} "
          f"{l:12.4e} {n:4d}  {why}", flush=True)
np.save(os.path.join(OUT_DIR, "summary.npy"), np.array(summary, dtype=object), allow_pickle=True)
print("\ndone.", flush=True)
