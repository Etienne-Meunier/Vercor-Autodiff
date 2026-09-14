"""Report 18: how wide is the basin around the true parameters?

The fine grid (dc_eps = 0.025) shows the loss sitting on a rough plateau of
~2e-5 right up to the truth, then collapsing to 1.4e-32 exactly on it. Two
readings fit that:

  (a) a smooth basin narrower than the 0.025 sampling, or
  (b) a genuine noise floor with an exact-cancellation needle at truth,
      because at the true parameters the rollout is bit-identical to the target.

They have opposite consequences. Under (a) a finer optimizer could still get
there; under (b) no gradient method can, because below the floor there is no
usable descent direction.

This walks c_eps toward 0.7 at fixed c_k = 0.1 on a log-spaced ladder down to
1e-5, and reports how the loss scales. A smooth basin gives loss ~ (dc_eps)^2
(a 100x drop per 10x step); a noise floor gives a flat value until the offset
underflows to zero.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build, make_kernels

import jax
import jax.numpy as jnp
import numpy as np

OUT_DIR = os.environ.get(
    "REPORT18_OUT_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-18")
)
os.makedirs(OUT_DIR, exist_ok=True)

OFFSETS = [0.025, 0.0125, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 1e-5]

cpl, initial_state, grid = build()
strat, make_loss = make_kernels(cpl, initial_state, grid)
strat_jit = jax.jit(strat)

target = jax.lax.stop_gradient(strat_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
loss_jit = jax.jit(make_loss(target))

print(f"c_k fixed at {TRUE_C_K}; walking c_eps toward {TRUE_C_EPS} (N_STEPS={N_STEPS})", flush=True)
print(f"\n{'d(c_eps)':>10s} {'c_eps':>10s} {'loss':>14s} {'loss/d^2':>12s} {'ratio to prev':>14s}", flush=True)

rows = []
prev = None
for d in OFFSETS:
    t0 = time.time()
    val = float(loss_jit({"c_k": jnp.asarray(TRUE_C_K), "c_eps": jnp.asarray(TRUE_C_EPS + d)}))
    ratio = (val / prev) if prev else float("nan")
    print(f"{d:10.1e} {TRUE_C_EPS + d:10.6f} {val:14.6e} {val / d**2:12.4e} {ratio:14.4f}"
          f"   [{time.time() - t0:.0f}s]", flush=True)
    rows.append((d, val))
    prev = val
    np.save(os.path.join(OUT_DIR, "basin_width.npy"), np.array(rows))

print("\nA smooth quadratic basin would show loss/d^2 roughly constant and each", flush=True)
print("ratio-to-prev near 0.25 when d halves. A noise floor shows loss roughly", flush=True)
print("constant and ratio-to-prev near 1.0.", flush=True)
print("done.", flush=True)
