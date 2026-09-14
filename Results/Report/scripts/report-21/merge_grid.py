"""Assemble `landscape_grid.py`'s per-job chunks into one loss grid.

Runs locally -- reads only .npy. Reports any node still missing rather than
silently writing a grid with holes in it.
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIG, "report-21-paper")

ck = np.load(os.path.join(OUT_DIR, "ck_grid.npy"))
ceps = np.load(os.path.join(OUT_DIR, "ceps_grid.npy"))
grid = np.full((ck.size, ceps.size), np.nan)

pattern = re.compile(r"^loss_grid_chunk_(\d+)_(\d+)\.npy$")
for name in sorted(os.listdir(OUT_DIR)):
    m = pattern.match(name)
    if not m:
        continue
    i0, i1 = int(m.group(1)), int(m.group(2))
    chunk = np.load(os.path.join(OUT_DIR, name))
    grid[i0:i1] = chunk
    print(f"{name}: columns [{i0}, {i1}), {int(np.isfinite(chunk).sum())} / {chunk.size} nodes")

missing = int(np.isnan(grid).sum())
print(f"\nmerged grid {grid.shape}: {grid.size - missing} nodes, {missing} missing")
if missing:
    idx = np.argwhere(np.isnan(grid))
    print("  missing (i, j): " + ", ".join(f"({i},{j})" for i, j in idx[:20])
          + (" ..." if missing > 20 else ""))

path = os.path.join(OUT_DIR, "loss_grid.npy")
np.save(path, grid)
print(f"saved {path}")
i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
print(f"argmin cell: c_k={ck[i]:.4f}, c_eps={ceps[j]:.4f}, loss={grid[i, j]:.6e} (true: 0.1, 0.7)")
