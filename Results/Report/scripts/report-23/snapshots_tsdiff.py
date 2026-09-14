"""Report 23 snapshot fields: top-level density difference and per-column loss.

Re-runs the 10-day coupled rollout at four parameter settings -- the truth (which
defines the target), the initial guess, the lowest-loss iterate and the final
iterate -- and saves, for each:

    d12_<label>.npy    mean over days 8-10 of prho[-2] - prho[-1]  (kg/m^3)
    ts_<label>.npy     the four tsdiff_avg channels (nx, ny, 4)
    colloss_<label>.npy  that run's loss per column: sum_c w_c (x_c - t_c)^2

`d12` is the summariser of what both losses actually see: report-23 shows the MLD
equals -35 - 0.9/d12 in 95% of columns, and the fitted T/S channels are the T and
S structure that sets it. The per-column loss is the fitted loss itself, mapped.

Same selection rule as report-21's snapshots.py: row k+1 of the trajectory holds
the parameters produced AFTER iteration k alongside the loss evaluated BEFORE it,
so the lowest-loss iterate's parameters are read from row k, not k+1.

    REPORT21_OUT_DIR=.../report-23-tsdiff-paper200/start_3 python snapshots_tsdiff.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-21"))

import jax
import jax.numpy as jnp
import numpy as np

from common21 import (
    AVG_WINDOW, INIT_C_EPS, INIT_C_K, N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K,
    build, loss_weights, make_payloads_fn, ts_diff_of_payload,
)

OUT_DIR = os.environ.get(
    "REPORT21_OUT_DIR",
    os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-23-tsdiff-paper200", "start_3"),
)

traj = np.load(os.path.join(OUT_DIR, "trajectory.npy"))
losses = traj[1:, 2]
best_k = int(np.nanargmin(losses))
BEST = (float(traj[best_k, 0]), float(traj[best_k, 1]))
FINAL = (float(traj[-1, 0]), float(traj[-1, 1]))
print(f"{traj.shape[0] - 1} iterations; start (c_k={INIT_C_K}, c_eps={INIT_C_EPS})", flush=True)
print(f"  best loss {np.nanmin(losses):.5e} at iteration {best_k}: "
      f"c_k={BEST[0]:.4f}, c_eps={BEST[1]:.4f}", flush=True)
print(f"  final iterate: c_k={FINAL[0]:.4f}, c_eps={FINAL[1]:.4f}", flush=True)

couplers, initial_state, grid = build(segmented=True)
payloads_fn = make_payloads_fn(couplers, initial_state)
surf = grid["surf"]
m2 = grid["maskT"][:, :, surf - 1]
m3 = grid["maskT"][:, :, surf - 2]
VALID = np.stack([m2, m3, m2, m3], axis=-1)


@jax.jit
def diagnostics(c_k, c_eps):
    """(four tsdiff channels, top-level density difference), both day-8-10 means."""
    ps = payloads_fn(c_k, c_eps)[-AVG_WINDOW:]
    ts = sum(ts_diff_of_payload(p) for p in ps) / len(ps)
    d12 = sum(p.variables.prho[2:-2, 2:-2, -2] - p.variables.prho[2:-2, 2:-2, -1] for p in ps) / len(ps)
    return ts, d12


fields = {}
for label, (a, b) in (("target", (TRUE_C_K, TRUE_C_EPS)),
                      ("init", (INIT_C_K, INIT_C_EPS)),
                      ("best", BEST),
                      ("final", FINAL)):
    t0 = time.time()
    ts, d12 = diagnostics(jnp.asarray(a), jnp.asarray(b))
    fields[label] = (np.asarray(ts), np.asarray(d12))
    print(f"  {label:>6s} (c_k={a:.4f}, c_eps={b:.4f}): {time.time() - t0:.1f}s", flush=True)

TARGET = fields["target"][0]
W = np.asarray(loss_weights("tsdiff_avg", TARGET, VALID))
print(f"loss weights: {W}", flush=True)

for label, (ts, d12) in fields.items():
    sq = np.where(VALID, ts - TARGET, 0.0) ** 2 * W
    colloss = sq.sum(axis=-1)
    np.save(os.path.join(OUT_DIR, f"ts_{label}.npy"), ts)
    np.save(os.path.join(OUT_DIR, f"d12_{label}.npy"), d12)
    np.save(os.path.join(OUT_DIR, f"colloss_{label}.npy"), colloss)
    print(f"  {label:>6s}: total loss {colloss.sum():.6e}, "
          f"d12 range {d12[m2].min():.4f} .. {d12[m2].max():.4f} kg/m^3", flush=True)

np.save(os.path.join(OUT_DIR, "d12_mask.npy"), m2)
np.save(os.path.join(OUT_DIR, "ts_valid.npy"), VALID)
np.save(os.path.join(OUT_DIR, "ts_weights.npy"), W)
np.save(os.path.join(OUT_DIR, "longitude.npy"), grid["longitude"])
np.save(os.path.join(OUT_DIR, "latitude.npy"), grid["latitude"])
np.save(os.path.join(OUT_DIR, "snapshot_params.npy"),
        np.array([[INIT_C_K, INIT_C_EPS], list(BEST), list(FINAL), [TRUE_C_K, TRUE_C_EPS]], float))
print("done.", flush=True)
