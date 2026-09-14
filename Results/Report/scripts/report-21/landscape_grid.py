"""Report 21 paper figure: loss landscape of the averaged-MLD observable.

Forward-only scan of `sum((mld_avg - target)^2)` over a grid in (c_k, c_eps) --
no gradient, so each node is one 10-day coupled rollout. Same ranges as report-8
and report-18 in `c_k`, extended in `c_eps` to 1.05 so that the multistart's
pre-registered corners (c_eps = 1.00) sit inside the map.

The scan is split by `c_k` column so several jobs can share it:

    REPORT21_GRID_I0=0 REPORT21_GRID_I1=4 python landscape_grid.py

writes `loss_grid_chunk_00_04.npy` holding columns [0, 4). `merge_grid.py`
assembles the chunks into `loss_grid.npy`. Each chunk checkpoints after every
node, so a job that dies resumes where it stopped rather than restarting.

Axis convention: `loss_grid[i, j]` with i over c_k and j over c_eps, matching
report-18 (report-8 stores the transpose).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from common21 import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build, loss_weights, make_observable

OBS = os.environ.get("REPORT21_OBS", "mld_avg")
OUT_DIR = os.environ.get(
    "REPORT21_PAPER_DIR", os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-21-paper")
)
os.makedirs(OUT_DIR, exist_ok=True)

CK_RANGE = (0.02, 0.18)
CEPS_RANGE = (0.25, 1.05)
GRID_N = int(os.environ.get("REPORT21_GRID_N", 16))

ck_grid = np.linspace(*CK_RANGE, GRID_N)
ceps_grid = np.linspace(*CEPS_RANGE, GRID_N)
I0 = int(os.environ.get("REPORT21_GRID_I0", 0))
I1 = int(os.environ.get("REPORT21_GRID_I1", GRID_N))
CHUNK = os.path.join(OUT_DIR, f"loss_grid_chunk_{I0:02d}_{I1:02d}.npy")
# per-term losses (one per observable channel), so each term's landscape can be drawn
COMP = os.path.join(OUT_DIR, f"losscomp_grid_chunk_{I0:02d}_{I1:02d}.npy")

np.save(os.path.join(OUT_DIR, "ck_grid.npy"), ck_grid)
np.save(os.path.join(OUT_DIR, "ceps_grid.npy"), ceps_grid)

print(f"observable={OBS}, N_STEPS={N_STEPS}, grid {GRID_N}x{GRID_N}", flush=True)
print(f"  c_k    {CK_RANGE[0]} .. {CK_RANGE[1]}", flush=True)
print(f"  c_eps  {CEPS_RANGE[0]} .. {CEPS_RANGE[1]}", flush=True)
print(f"  this chunk: c_k columns [{I0}, {I1}) -> {os.path.basename(CHUNK)}", flush=True)

couplers, initial_state, grid = build(segmented=OBS.endswith("_avg"))
observable = make_observable(OBS, couplers, initial_state, grid)
obs_jit = jax.jit(observable)

print(f"\ngenerating target (true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...", flush=True)
t0 = time.time()
target, target_valid = obs_jit(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target = jax.lax.stop_gradient(target)
TARGET_VALID = jnp.asarray(np.asarray(target_valid))
print(f"  done ({time.time() - t0:.1f}s incl. compile); "
      f"{int(np.asarray(TARGET_VALID).sum())} cells in loss", flush=True)
W = loss_weights(OBS, target, TARGET_VALID)
print(f"loss weights: {np.asarray(W)}", flush=True)
N_TERMS = int(np.asarray(target).shape[-1]) if np.asarray(target).ndim == 3 else 1


@jax.jit
def loss_terms(c_k, c_eps):
    """The same loss fit.py descends (masked, weighted SSE on the frozen target
    domain), split per observable channel; the loss is the sum of the terms."""
    field, valid = observable(c_k, c_eps)
    sq = W * jnp.where(TARGET_VALID & valid, field - target, 0.0) ** 2
    return jnp.sum(sq, axis=(0, 1)) if sq.ndim == 3 else jnp.sum(sq)[None]


if os.path.exists(CHUNK):
    chunk = np.load(CHUNK)
    print(f"resuming: {int(np.isfinite(chunk).sum())} / {chunk.size} nodes already done", flush=True)
else:
    chunk = np.full((I1 - I0, GRID_N), np.nan)
comp = np.load(COMP) if os.path.exists(COMP) else np.full((I1 - I0, GRID_N, N_TERMS), np.nan)

t_grid = time.time()
for i in range(I0, I1):
    for j, ce in enumerate(ceps_grid):
        if np.isfinite(chunk[i - I0, j]):
            continue
        t0 = time.time()
        terms = np.asarray(loss_terms(jnp.asarray(ck_grid[i]), jnp.asarray(ce)))
        comp[i - I0, j] = terms
        chunk[i - I0, j] = float(terms.sum())
        print(f"  ({i:2d},{j:2d}) c_k={ck_grid[i]:.4f} c_eps={ce:.4f}  "
              f"loss={chunk[i - I0, j]:.6e}  terms={np.array2string(terms, precision=4)}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        np.save(COMP, comp)
        np.save(CHUNK, chunk)
print(f"\nchunk wall time: {time.time() - t_grid:.1f}s", flush=True)
print(f"chunk min loss {np.nanmin(chunk):.6e}, max {np.nanmax(chunk):.6e}", flush=True)
print("done.", flush=True)
