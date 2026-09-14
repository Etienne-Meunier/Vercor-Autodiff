"""Report 23 figure 3: what the T/S loss measures, and where it is wrong.

Reads only the .npy written by snapshots_tsdiff.py -- no model, no cluster:

    figures/report-23-tsdiff-paper200/start_3/{d12,colloss}_{target,init,final}.npy

Left    the target's top-level density difference, mean over days 8-10:
            d12 = prho[-2] - prho[-1]
        which is what both this loss and report-22's MLD actually respond to
        (report-23: mld = -35 - 0.9/d12 in 95% of columns).
Middle  each column's contribution to the fitted loss BEFORE calibration
Right   the same AFTER calibration, at start 3's converged parameters

The two loss panels share one logarithmic colour scale, so the colours compare;
a linear scale renders the "after" panel blank. Run with the `veros` env: cartopy
needs shapely >= 2.0.5 under numpy 2.

    ~/miniforge3/envs/veros/bin/python figure_snapshot.py [WHICH]   # final (default) | best
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
PAPER = os.path.join(FIG, "report-23-tsdiff-paper200")
SNAP = os.path.join(PAPER, "start_3")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "final"

lon = np.load(os.path.join(SNAP, "longitude.npy"))
lat = np.load(os.path.join(SNAP, "latitude.npy"))
mask = np.load(os.path.join(SNAP, "d12_mask.npy"))
params = np.load(os.path.join(SNAP, "snapshot_params.npy"))
(init_ck, init_ce), (best_ck, best_ce), (fit_ck, fit_ce), (true_ck, true_ce) = params
if WHICH == "best":
    fit_ck, fit_ce = best_ck, best_ce


def masked(name):
    return np.where(mask, np.load(os.path.join(SNAP, f"{name}.npy")), np.nan)


d12 = masked("d12_target")
loss_before, loss_after = masked("colloss_init"), masked(f"colloss_{WHICH}")
tot_b, tot_a = np.nansum(loss_before), np.nansum(loss_after)
n = int(mask.sum())
print(f"{n} columns; total loss before {tot_b:.4e}, after ({WHICH}) {tot_a:.4e} "
      f"({tot_b / tot_a:.1f}x reduction)")
print(f"target d12: {np.nanmin(d12):.4f} .. {np.nanmax(d12):.4f} kg/m^3 "
      f"(median {np.nanmedian(d12):.4f})")
# Where the residual concentrates: the top 1% of columns by loss.
for lab, f in (("before", loss_before), ("after", loss_after)):
    q = np.nanpercentile(f, 99)
    print(f"  {lab}: top 1% of columns carry {np.nansum(f[f >= q]) / np.nansum(f) * 100:.0f}% of the loss")

# Shared log scale for the two loss panels; floor at 1e-4 of the "before" 99th
# percentile so near-zero columns do not stretch the scale over 15 decades.
hi = float(np.nanpercentile(loss_before, 99.5))
lo = hi * 1e-4
t_lo, t_hi = (float(np.nanpercentile(d12, q)) for q in (2, 98))

PROJ = ccrs.Robinson(central_longitude=200)
DATA_CRS = ccrs.PlateCarree()
LAND_COLOR, NODATA_COLOR = "#cfcfcf", "#f2efe9"
plt.rcParams.update({"font.size": 11})

fig = plt.figure(figsize=(16, 4.2))
gs = fig.add_gridspec(1, 3, wspace=0.04, left=0.02, right=0.98, top=0.84, bottom=0.2)
ax_t, ax_b, ax_a = (fig.add_subplot(gs[0, i], projection=PROJ) for i in range(3))


def titled(ax, title, sub):
    ax.set_title(sub, fontsize=10, pad=4)
    ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 20),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")


im_t = ax_t.pcolormesh(lon, lat, d12.T, cmap="viridis", shading="auto", transform=DATA_CRS,
                       vmin=t_lo, vmax=t_hi, rasterized=True)
titled(ax_t, "Target top-level density difference",
       f"$c_k$ = {true_ck},  $c_\\epsilon$ = {true_ce}  (truth)")

for ax, f, title, ck_, ce_, tot in (
    (ax_b, loss_before, "Loss per column before", init_ck, init_ce, tot_b),
    (ax_a, loss_after, "Loss per column after", fit_ck, fit_ce, tot_a),
):
    im_l = ax.pcolormesh(lon, lat, np.maximum(f, lo).T, cmap="magma_r", shading="auto",
                         transform=DATA_CRS, norm=LogNorm(vmin=lo, vmax=hi), rasterized=True)
    titled(ax, title, f"$c_k$ = {ck_:.4f},  $c_\\epsilon$ = {ce_:.4f}   |   total {tot:.2e}")

for ax in (ax_t, ax_b, ax_a):
    ax.set_global()
    ax.set_facecolor(NODATA_COLOR)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555555", linewidth=0.4, zorder=3)
    ax.gridlines(color="#7f7f7f", linewidth=0.3, alpha=0.5, linestyle="--",
                 xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
    ax.spines["geo"].set_linewidth(0.6)

fig.canvas.draw()


def cbar_below(left, right, frac, im, label, extend):
    p0, p1 = left.get_position(), right.get_position()
    w = (p1.x1 - p0.x0) * frac
    cax = fig.add_axes([0.5 * (p0.x0 + p1.x1) - w / 2, p0.y0 - 0.075, w, 0.035])
    fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend).set_label(label)


cbar_below(ax_t, ax_t, 0.7, im_t,
           r"$\rho_\theta[-2] - \rho_\theta[-1]$ (kg m$^{-3}$)", "both")
cbar_below(ax_b, ax_a, 0.45, im_l, "loss contribution per column (variance-normalised)", "min")

path = os.path.join(PAPER, f"figure3_tsdiff_snapshot{'' if WHICH == 'final' else '_best'}.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
print(f"saved {path} (+ .pdf)")
