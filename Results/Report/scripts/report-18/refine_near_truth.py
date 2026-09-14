"""Report 18: fine grid centred on the true parameters.

The 12x12 scan samples c_k at 0.0927 and 0.1073 and c_eps at 0.6318 and 0.6955 --
it never evaluates the true (0.1, 0.7). So it cannot show that the minimum is
where it must be, and the star necessarily plots off the darkest sampled cell.

This evaluates the loss ON the truth (where it is exactly 0 by construction --
the target is generated from those parameters) and on a fine grid around it, at a
resolution that resolves the diagonal valley: dc_k = 0.005 against the coarse
scan's 0.0145, dc_eps = 0.025 against 0.0636.

Forward-only, no gradients. ~30 evaluations at roughly 40 s each on g5k.
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

CK_FINE = np.round(np.arange(0.085, 0.1151, 0.005), 4)      # 0.085 .. 0.115
CEPS_FINE = np.round(np.arange(0.600, 0.8001, 0.025), 4)    # 0.600 .. 0.800

cpl, initial_state, grid = build()
strat, make_loss = make_kernels(cpl, initial_state, grid)
strat_jit = jax.jit(strat)

print(f"target from true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS})", flush=True)
target = jax.lax.stop_gradient(strat_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
loss_jit = jax.jit(make_loss(target))


def loss_at(a, b):
    return float(loss_jit({"c_k": jnp.asarray(a), "c_eps": jnp.asarray(b)}))


# The defining check: the loss at the parameters the target was generated from.
t0 = time.time()
l_truth = loss_at(TRUE_C_K, TRUE_C_EPS)
print(f"\nLOSS AT TRUTH ({TRUE_C_K}, {TRUE_C_EPS}) = {l_truth:.6e}   "
      f"({'as expected: exactly zero' if l_truth == 0.0 else 'NOT ZERO -- investigate'})  "
      f"[{time.time() - t0:.1f}s]", flush=True)

PATH = os.path.join(OUT_DIR, "fine_grid.npy")
if os.path.exists(PATH) and np.load(PATH).shape == (len(CK_FINE), len(CEPS_FINE)):
    fine = np.load(PATH)
else:
    fine = np.full((len(CK_FINE), len(CEPS_FINE)), np.nan)

print(f"\nfine grid {len(CK_FINE)}x{len(CEPS_FINE)} = {fine.size} forward evals "
      f"(dc_k={CK_FINE[1] - CK_FINE[0]:.4f}, dc_eps={CEPS_FINE[1] - CEPS_FINE[0]:.4f})", flush=True)
for i, a in enumerate(CK_FINE):
    for j, b in enumerate(CEPS_FINE):
        if np.isfinite(fine[i, j]):
            continue
        t0 = time.time()
        fine[i, j] = loss_at(a, b)
        print(f"  c_k={a:.4f} c_eps={b:.4f}  loss={fine[i, j]:.6e}  ({time.time() - t0:.1f}s)", flush=True)
        np.save(PATH, fine)
np.save(os.path.join(OUT_DIR, "fine_ck.npy"), CK_FINE)
np.save(os.path.join(OUT_DIR, "fine_ceps.npy"), CEPS_FINE)
np.save(os.path.join(OUT_DIR, "loss_at_truth.npy"), np.array([l_truth]))

i, j = np.unravel_index(np.nanargmin(fine), fine.shape)
print(f"\nfine-grid minimum {np.nanmin(fine):.6e} at c_k={CK_FINE[i]:.4f}, c_eps={CEPS_FINE[j]:.4f}", flush=True)
print(f"loss at truth = {l_truth:.6e}", flush=True)
print("done.", flush=True)
