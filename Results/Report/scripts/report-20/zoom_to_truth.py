"""Report 20: zoom in on the approach to the truth from a converged GN endpoint.

Gauss-Newton reaches c_k=0.0999, c_eps=0.7003 -- 0.1% parameter error -- with a
loss of 1.0e-2, while the loss AT the truth is 3e-14. Nine orders of magnitude
separate them across 0.1% of parameter space. Two readings:

  (a) a smooth bowl the optimizer simply stopped short of, in which case the loss
      falls steadily as the gap closes, or
  (b) the rugged floor of report-18/19 -- the loss sits at ~1e-2 until the
      parameters are bit-identical, and the 3e-14 is exact cancellation with no
      approach to it.

Log-spaced fractions along the segment from the GN endpoint to the truth
distinguish them.
"""

import os
import sys
import time

os.environ.setdefault("REPORT18_N_STEPS", "10")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report-18"))

from common import N_STEPS, REPO_ROOT, TRUE_C_EPS, TRUE_C_K, build  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from vercor.setups._external.veros_state import set_veros_variable  # noqa: E402

OUT_DIR = os.environ.get("REPORT20_OUT_DIR",
                         os.path.join(REPO_ROOT, "Results", "Report", "figures", "report-20-gn-multistart"))
os.makedirs(OUT_DIR, exist_ok=True)

FROM_C_K = float(os.environ.get("REPORT20_FROM_CK", 0.0999))
FROM_C_EPS = float(os.environ.get("REPORT20_FROM_CEPS", 0.7003))
FRACS = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]

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


TARGET = jax.lax.stop_gradient(jax.jit(surface_temp)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))
loss_jit = jax.jit(lambda a, b: jnp.sum(jnp.where(SMASK, surface_temp(a, b) - TARGET, 0.0) ** 2))

print(f"from (c_k={FROM_C_K}, c_eps={FROM_C_EPS}) to the truth ({TRUE_C_K}, {TRUE_C_EPS}), "
      f"N_STEPS={N_STEPS}", flush=True)
print(f"\n{'fraction':>10s} {'gap in c_k':>12s} {'gap in c_eps':>13s} {'loss':>13s}", flush=True)
rows = []
for f in FRACS:
    a = FROM_C_K + f * (TRUE_C_K - FROM_C_K)
    b = FROM_C_EPS + f * (TRUE_C_EPS - FROM_C_EPS)
    t0 = time.time()
    v = float(loss_jit(jnp.asarray(a), jnp.asarray(b)))
    rows.append((f, a, b, v))
    print(f"{f:10.4f} {TRUE_C_K - a:12.2e} {TRUE_C_EPS - b:13.2e} {v:13.5e}   [{time.time() - t0:.0f}s]",
          flush=True)

np.save(os.path.join(OUT_DIR, "zoom_to_truth.npy"), np.array(rows))
L = np.array([r[3] for r in rows])
print(f"\nloss falls {L[0]:.3e} -> {L[-1]:.3e}", flush=True)
print(f"at 99.99% of the way there the loss is still {L[-2]:.3e}, "
      f"{L[-2] / max(L[-1], 1e-300):.1e}x the value at the truth", flush=True)
print("A smooth bowl would show the loss falling steadily with the gap; a rugged", flush=True)
print("floor shows it flat until the parameters coincide exactly.", flush=True)
print("done.", flush=True)
