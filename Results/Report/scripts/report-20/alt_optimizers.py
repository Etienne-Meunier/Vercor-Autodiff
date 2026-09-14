"""Report 20: Gauss-Newton and L-BFGS on the same N=10 surface-temperature fit.

Report-20's adam run stops at c_eps = 0.535-0.562 not because of a local minimum
(the loss at truth is 3e-14 and the straight path there never rises above 1.43x)
but because of a narrow diagonal valley: dL/dc_eps points the WRONG way along the
whole path, so first-order descent slides along the valley instead of down it.

Two optimizers that should handle that, both from the identical start and loss:

  gn     Gauss-Newton with a forward-mode Jacobian. The residual is the masked
         surface-temperature difference, so with 2 parameters two JVPs give the
         exact Jacobian J (n_cells x 2) and the step is -(J^T J)^-1 J^T r. J^T J
         is the same 2x2 matrix report-17 computes; inverting it is exactly the
         correction the valley needs, discovered from local information only --
         no knowledge of the true parameters, unlike hard-coding the valley slope.
         Levenberg-Marquardt damping makes it robust to the loss's ruggedness.

  lbfgs  scipy's L-BFGS-B on the same value_and_grad. Operates in parameter space
         (2 dimensions), so the size of the dynamical system is irrelevant to it;
         the only extra cost over adam is line-search function evaluations.

Env: REPORT20_MODE = gn | lbfgs
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")
os.environ.setdefault("REPORT18_N_ITERS", "30")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import (  # noqa: E402
    INIT_C_EPS, INIT_C_K, N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

MODE = os.environ.get("REPORT20_MODE", "gn").lower()
MAX_ITERS = int(os.environ.get("REPORT20_MAX_ITERS", 25))
OUT_DIR = os.environ.get("REPORT20_OUT_DIR",
                         os.path.join(REPO_ROOT, "Results", "Report", "figures", f"report-20-{MODE}"))
os.makedirs(OUT_DIR, exist_ok=True)
print(f"mode={MODE}  N_STEPS={N_STEPS}  max_iters={MAX_ITERS}  out={OUT_DIR}", flush=True)

cpl, initial_state, grid = build()
MASK_NP = grid["maskT"]
SURF = MASK_NP.shape[2] - 1
SMASK = jnp.asarray(MASK_NP[:, :, SURF])


def surface_temp(c_k, c_eps):
    cs = initial_state._component_state("OCN")
    p = set_veros_variable(cs.payload, "c_k", c_k)
    p = set_veros_variable(p, "c_eps", c_eps)
    st = initial_state._with_component_state("OCN", cs.with_payload(p))
    pay = cpl.run(st, output=None)._component_state("OCN").payload
    return pay.variables.temp[2:-2, 2:-2, SURF, pay.variables.tau]


print(f"generating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...", flush=True)
t0 = time.time()
TARGET = jax.lax.stop_gradient(jax.jit(surface_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
print(f"  done ({time.time() - t0:.0f}s)", flush=True)


def residual_vec(theta):
    """Masked residual field, flattened. Land cells contribute exactly zero to
    both the residual and the Jacobian, so they drop out of the normal equations."""
    t = surface_temp(theta[0], theta[1])
    return jnp.where(SMASK, t - TARGET, 0.0).ravel()


def loss_of(theta):
    return jnp.sum(residual_vec(theta) ** 2)


trajectory = [(INIT_C_K, INIT_C_EPS, None)]
TRAJ = os.path.join(OUT_DIR, "trajectory.npy")


def record(ck, ce, loss):
    trajectory.append((float(ck), float(ce), float(loss)))
    np.save(TRAJ, np.array([[a, b, c if c is not None else np.nan] for a, b, c in trajectory]))


def report(tag, ck, ce, loss, extra=""):
    print(f"{tag:>5s}  loss={float(loss):13.6e}  c_k={float(ck):8.4f} ({100 * abs(ck - TRUE_C_K) / TRUE_C_K:5.1f}%)  "
          f"c_eps={float(ce):8.4f} ({100 * abs(ce - TRUE_C_EPS) / TRUE_C_EPS:5.1f}%)  {extra}", flush=True)


if MODE == "gn":
    def residual_pair(a, b):
        t = surface_temp(a, b)
        return jnp.where(SMASK, t - TARGET, 0.0).ravel()

    def jac_and_residual(a, b):
        """Jacobian columns from two explicit JVPs.

        NOTE: jax.jacfwd on this residual returns inf columns -- its internal
        vmap over the tangent basis hits something in the coupled model that two
        separate jvp calls do not. Verified at the start point: jacfwd column
        norms are (inf, inf) while the explicit JVPs give (41.7, 10.6) and
        cond(JtJ) = 24. Same cost either way: two tangents.
        """
        _, t_k = jax.jvp(lambda x: residual_pair(x, b), (a,), (jnp.ones_like(a),))
        r, t_e = jax.jvp(lambda y: residual_pair(a, y), (b,), (jnp.ones_like(b),))
        return r, jnp.stack([t_k, t_e], axis=1)

    jac_fn = jax.jit(jac_and_residual)
    res_fn = jax.jit(residual_vec)

    theta = np.array([INIT_C_K, INIT_C_EPS])
    r = np.asarray(res_fn(jnp.asarray(theta)))
    loss = float(r @ r)
    report("init", theta[0], theta[1], loss)
    lam = 1e-3  # Levenberg-Marquardt damping

    for it in range(MAX_ITERS):
        t0 = time.time()
        _r, _J = jac_fn(jnp.asarray(theta[0]), jnp.asarray(theta[1]))
        J = np.asarray(_J)                                   # (n_cells, 2), two JVPs
        JTJ, JTr = J.T @ J, J.T @ r
        w = np.linalg.eigvalsh(JTJ)
        cond = float(w[-1] / w[0]) if w[0] > 0 else np.inf

        accepted = False
        for _ in range(6):                                   # damping search, no new Jacobian
            # LM damping. Scaling by diag(JTJ) alone is singular whenever a
            # parameter has no effect at all (c_eps at N<=2, per report-17), so
            # the damping matrix carries an absolute floor, and the solve falls
            # back to a least-squares solution if it is still rank-deficient.
            scale = np.maximum(np.diag(JTJ), 1e-12 * max(float(np.max(np.abs(JTJ))), 1.0))
            A = JTJ + lam * np.diag(scale)
            try:
                step = -np.linalg.solve(A, JTr)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A, JTr, rcond=None)[0]
            cand = theta + step
            cand[0] = max(cand[0], 1e-4)
            cand[1] = max(cand[1], 1e-4)
            r_new = np.asarray(res_fn(jnp.asarray(cand)))
            loss_new = float(r_new @ r_new)
            if loss_new < loss:
                theta, r, loss = cand, r_new, loss_new
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 5.0
        record(theta[0], theta[1], loss)
        report(f"{it}", theta[0], theta[1], loss,
               f"cond(JtJ)={cond:.2e} lam={lam:.1e} {'' if accepted else 'NO STEP ACCEPTED'} "
               f"[{time.time() - t0:.0f}s]")
        if not accepted:
            print("  damping search exhausted -- stopping", flush=True)
            break
        if np.linalg.norm(step) < 1e-8:
            print("  step below tolerance -- converged", flush=True)
            break

elif MODE == "lbfgs":
    from scipy.optimize import minimize

    vg = jax.jit(jax.value_and_grad(loss_of))
    calls = {"n": 0}

    def fun(x):
        t0 = time.time()
        v, g = vg(jnp.asarray(x))
        calls["n"] += 1
        v = float(v)
        g = np.asarray(g, dtype=np.float64)
        print(f"  eval {calls['n']:3d}  loss={v:13.6e}  c_k={x[0]:8.4f}  c_eps={x[1]:8.4f}  "
              f"grad=({g[0]:.3e}, {g[1]:.3e})  [{time.time() - t0:.0f}s]", flush=True)
        record(x[0], x[1], v)
        return v, g

    res = minimize(fun, np.array([INIT_C_K, INIT_C_EPS]), jac=True, method="L-BFGS-B",
                   bounds=[(1e-4, 1.0), (1e-4, 3.0)],
                   options={"maxiter": MAX_ITERS, "maxfun": 3 * MAX_ITERS, "ftol": 1e-14, "gtol": 1e-12})
    print(f"\nscipy status: {res.message}  (nit={res.nit}, nfev={res.nfev})", flush=True)
    theta, loss = res.x, float(res.fun)

else:
    raise SystemExit(f"unknown REPORT20_MODE={MODE!r}; use gn or lbfgs")

print("", flush=True)
report("FINAL", theta[0], theta[1], loss)
print(f"\nfor comparison, adam at N=10 (100 iters, clipped): "
      f"best-loss c_k=0.0807 (19.3%), c_eps=0.5353 (23.5%), loss=4.585e-02", flush=True)
print("done.", flush=True)
