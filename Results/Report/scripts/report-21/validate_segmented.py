"""Check that report-21's chained rollout reproduces the monolithic one.

The averaged observables need the day-8/9/10 states, so the 10-day rollout is run
as three chained couplers (days 0-8, 8-9, 9-10) instead of one 10-step coupler.
That is only legitimate if handing a returned state to a coupler started at the
date the previous one ended reproduces the uninterrupted run -- if, say, a
component re-initialised its forcing from its own clock start, the chain would
silently integrate the wrong days.

So: roll out both ways at the same parameters and compare the day-10 surface
temperature and MLD. Expected agreement is roundoff (x64, so ~1e-12 relative);
anything larger invalidates the averaged runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import common21
from common21 import INIT_C_EPS, INIT_C_K, build, make_observable

for kind in ("temp", "mld"):
    print(f"\n=== {kind} ===", flush=True)
    fields = {}
    for segmented in (False, True):
        couplers, initial_state, grid = build(segmented=segmented)
        obs = jax.jit(make_observable(kind, couplers, initial_state, grid))
        t0 = time.time()
        field, valid = obs(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS))
        fields[segmented] = (np.asarray(field), np.asarray(valid))
        print(f"  {'chained' if segmented else 'monolithic'}: {time.time() - t0:.1f}s", flush=True)

    (mono, mono_valid), (seg, seg_valid) = fields[False], fields[True]
    mask = mono_valid & seg_valid
    d = np.abs(mono - seg)[mask]
    scale = np.abs(mono[mask]).mean()
    print(f"  cells compared: {int(mask.sum())} (mask mismatch: {int((mono_valid != seg_valid).sum())})", flush=True)
    print(f"  max |difference| = {d.max():.3e}, mean field magnitude {scale:.3e}, "
          f"relative {d.max() / scale:.3e}", flush=True)
    print("  VERDICT: " + ("chained == monolithic to roundoff" if d.max() / scale < 1e-9
                           else "MISMATCH -- the chained rollout is NOT the same integration"), flush=True)
