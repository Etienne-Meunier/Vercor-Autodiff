"""Report 23: top-3 T/S-difference landscape against report-22's averaged-MLD one.

Reads only .npy, runs locally after the grid chunks are pulled back:

    python landscape_compare.py [TSDIFF_DIR] [MLD_DIR]

Panels, each normalised by its own lowest sampled node and drawn in log10:
  1. report-22's mld_avg loss
  2. tsdiff_avg loss (the one fit.py descends)
  3. its temperature terms only (T[-2]-T[-1], T[-3]-T[-1])
  4. its salinity terms only (S[-2]-S[-1], S[-3]-S[-1])

Local minima are nodes strictly below all 8 neighbours (edge nodes compare with
the neighbours they have) -- the same search report-22 quotes. Descent paths are
overlaid when `report-23-tsdiff-paper200/start_*/trajectory.npy` exist.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
TS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIG, "report-23-tsdiff-paper")
MLD_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(FIG, "report-21-paper")
PATH_DIR = os.path.join(FIG, "report-23-tsdiff-paper200")
TRUE_C_K, TRUE_C_EPS = 0.1, 0.7


def load_components(out_dir, shape):
    """Assemble losscomp_grid_chunk_*.npy into (n_ck, n_ceps, n_terms)."""
    comp = None
    for name in sorted(os.listdir(out_dir)):
        m = re.match(r"^losscomp_grid_chunk_(\d+)_(\d+)\.npy$", name)
        if not m:
            continue
        chunk = np.load(os.path.join(out_dir, name))
        if comp is None:
            comp = np.full(shape + (chunk.shape[-1],), np.nan)
        comp[int(m.group(1)):int(m.group(2))] = chunk
    return comp


def local_minima(grid):
    n0, n1 = grid.shape
    out = []
    for i in range(n0):
        for j in range(n1):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            nb = [grid[a, b] for a in range(max(i - 1, 0), min(i + 2, n0))
                  for b in range(max(j - 1, 0), min(j + 2, n1)) if (a, b) != (i, j)]
            if all(np.isfinite(x) and v < x for x in nb):
                out.append((i, j, v))
    return sorted(out, key=lambda t: t[2])


ck = np.load(os.path.join(TS_DIR, "ck_grid.npy"))
ceps = np.load(os.path.join(TS_DIR, "ceps_grid.npy"))
ts = np.load(os.path.join(TS_DIR, "loss_grid.npy"))
comp = load_components(TS_DIR, ts.shape)
mld = np.load(os.path.join(MLD_DIR, "loss_grid.npy"))
assert np.allclose(np.load(os.path.join(MLD_DIR, "ck_grid.npy")), ck), "grids differ in c_k"
assert np.allclose(np.load(os.path.join(MLD_DIR, "ceps_grid.npy")), ceps), "grids differ in c_eps"

panels = [("report-22: MLD, mean days 8-10", mld), ("top-3 T/S differences (fitted loss)", ts)]
if comp is not None and comp.shape[-1] == 4:
    panels += [("temperature terms only", comp[..., 0] + comp[..., 1]),
               ("salinity terms only", comp[..., 2] + comp[..., 3])]

for title, g in panels:
    print(f"\n{title}: {int(np.isfinite(g).sum())}/{g.size} nodes, "
          f"range {np.nanmin(g):.4e} .. {np.nanmax(g):.4e} ({np.nanmax(g) / np.nanmin(g):.1f}x)")
    for rank, (i, j, v) in enumerate(local_minima(g)):
        print(f"  local min {rank}: c_k={ck[i]:.4f} c_eps={ceps[j]:.4f} loss={v:.4e} "
              f"({v / np.nanmin(g):.2f}x the lowest node)")

def load_paths(path_dir):
    out = []
    for s in range(4):
        p = os.path.join(path_dir, f"start_{s}", "trajectory.npy")
        if os.path.exists(p):
            out.append((s, np.load(p)))
    return out


# each landscape carries the descents run on it: report-22's MLD paths on the MLD
# panel, this report's T/S paths on the other three
ts_paths = load_paths(PATH_DIR)
mld_paths = load_paths(os.path.join(FIG, "report-21-paper200"))

fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), sharey=True)
for k, (ax, (title, g)) in enumerate(zip(np.atleast_1d(axs), panels)):
    paths = mld_paths if k == 0 else ts_paths
    rel = np.log10(g / np.nanmin(g))
    im = ax.pcolormesh(ck, ceps, rel.T, shading="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10(loss / lowest node)")
    ax.contour(ck, ceps, rel.T, levels=np.arange(0.25, rel[np.isfinite(rel)].max(), 0.25),
               colors="white", linewidths=0.4, alpha=0.5)
    for rank, (i, j, _) in enumerate(local_minima(g)):
        ax.plot(ck[i], ceps[j], "o" if rank == 0 else "s", mfc="none", mec="red", ms=9, mew=1.5)
    for s, tr in paths:
        ax.plot(tr[:, 0], tr[:, 1], "-", color="white", lw=0.9)
        ax.plot(tr[0, 0], tr[0, 1], "o", color="white", ms=5)
        ax.plot(tr[-1, 0], tr[-1, 1], "x", color="orange", ms=8, mew=2)
        ax.annotate(str(s), (tr[0, 0], tr[0, 1]), color="white", fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=16, mec="black")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("c_k")
np.atleast_1d(axs)[0].set_ylabel("c_eps")
fig.suptitle("Report 23: loss landscapes (red circle = lowest local minimum, squares = other local minima)")
fig.tight_layout()
out = os.path.join(TS_DIR, "landscape_compare.png")
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
