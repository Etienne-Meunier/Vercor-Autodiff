"""The two paper figures for the averaged-MLD calibration demonstration.

Reads only .npy -- no model, no cluster -- so the figures can be restyled in
seconds without recomputing anything:

    figures/report-21-paper/loss_grid.npy, ck_grid.npy, ceps_grid.npy
        forward-only loss landscape from landscape_grid.py + merge_grid.py
    figures/report-21-paper200/start_{0..3}/trajectory.npy
        one descent per pre-registered start, 5 columns
        (c_k, c_eps, loss, dL/dc_k, dL/dc_eps); row k+1 holds the parameters
        produced after iteration k and the loss evaluated before it
    figures/report-21-paper200/start_0/field_*.npy, domain.npy, longitude.npy,
    latitude.npy, snapshot_params.npy
        the day-8-to-10 mean MLD at truth, at the initial guess and at the fitted
        parameters, plus the column domain common to all of them

Figure 1  loss landscape with the four descents on it, and their loss curves.
Figure 2  target mean MLD, and its error before and after assimilation.

The starts are the four corners of a box centred on the true parameters
(c_k = 0.1 +/- 50%, c_eps = 0.7 +/- 0.3), fixed before the runs were launched.
Start 0 is the canonical (0.05, 0.40) used throughout this series.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
# Which multistart set to draw. The 100-iteration set (report-21-paper) is the
# original budget inherited from report-20; the 200-iteration set doubles it after
# the 100-iteration runs were found to stop before converging -- start 0 reaches
# the true parameters only in the longer set. Both are kept so the two can be
# compared; set REPORT21_PAPER_SET=report-21-paper to draw the shorter one.
PAPER = os.path.join(FIG, os.environ.get("REPORT21_PAPER_SET", "report-21-paper200"))
# The loss grid was scanned once and lives with the 100-iteration set.
GRIDDIR = os.path.join(FIG, "report-21-paper")
# Fields for figure 2 sit beside the run whose fitted point they were rolled out at.
SNAP = os.environ.get("REPORT21_SNAP_DIR", os.path.join(PAPER, "start_0"))
os.makedirs(PAPER, exist_ok=True)

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
STARTS = [(0.05, 0.40), (0.15, 1.00), (0.05, 1.00), (0.15, 0.40)]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
SMOOTH_WINDOW = int(os.environ.get("REPORT21_LOSS_SMOOTH", 9))   # iterations
MAP_SIGMA = float(os.environ.get("REPORT21_MAP_SIGMA", 1.0))  # grid cells; 0 disables

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})


def smooth_loss(y, window=SMOOTH_WINDOW):
    """Rolling median of the loss in log space.

    Median, not mean: start 0 touches 9.3e-6 at a single iterate, five decades
    below its neighbours, and a mean would drag the smoothed curve down to an
    excursion no other iterate shares. Log space because the loss spans decades,
    so the smoothed curve is a geometric running centre. Edges use a shrinking
    window rather than padding, so the curve starts and ends on real data.
    """
    y = np.asarray(y, float)
    out = np.empty_like(y)
    half = window // 2
    logy = np.log10(np.maximum(y, np.finfo(float).tiny))
    for i in range(y.size):
        lo, hi = max(0, i - half), min(y.size, i + half + 1)
        out[i] = np.median(logy[lo:hi])
    # A median filter leaves visible stair-steps; a short moving average over the
    # already-robust curve softens them without letting the outlier back in.
    smoothed = np.empty_like(out)
    for i in range(out.size):
        lo, hi = max(0, i - 2), min(out.size, i + 3)
        smoothed[i] = out[lo:hi].mean()
    return 10.0 ** smoothed


def _convolve1d(a, w, axis, periodic):
    """Separable 1D convolution, wrapping in longitude and clamping in latitude."""
    r = (w.size - 1) // 2
    if periodic:
        return sum(w[i] * np.roll(a, r - i, axis=axis) for i in range(w.size))
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    b = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i in range(w.size):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + a.shape[axis])
        out = out + w[i] * b[tuple(sl)]
    return out


def smooth_map(field, sigma=MAP_SIGMA):
    """NaN-aware Gaussian smoothing, periodic in longitude.

    Cosmetic only -- every RMSE quoted in this figure is computed from the
    unsmoothed field. Land and undefined columns are NaN, so the data (with NaN
    set to 0) and the validity mask are convolved with the same kernel and then
    divided: each cell becomes a weighted mean over the cells that actually hold
    data, renormalised, which stops the coastline from bleeding ocean values
    outward or pulling coastal cells toward zero. The result is re-masked, so the
    field cannot grow into land.
    """
    if sigma <= 0:
        return field
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()

    valid = np.isfinite(field).astype(float)
    data = np.where(np.isfinite(field), field, 0.0)
    for axis, periodic in ((0, True), (1, False)):  # axis 0 is longitude
        data = _convolve1d(data, w, axis, periodic)
        valid = _convolve1d(valid, w, axis, periodic)
    out = np.where(valid > 1e-6, data / np.where(valid > 1e-6, valid, 1.0), np.nan)
    return np.where(np.isfinite(field), out, np.nan)


# --- Figure 1: landscape, paths, loss curves ---------------------------------

GRID_PATH = os.path.join(GRIDDIR, "loss_grid.npy")
HAVE_GRID = os.path.exists(GRID_PATH)
if not HAVE_GRID:
    print(f"WARNING: {GRID_PATH} missing (run merge_grid.py); skipping figure 1")
else:
    grid = np.load(GRID_PATH)
    ck = np.load(os.path.join(GRIDDIR, "ck_grid.npy"))
    ceps = np.load(os.path.join(GRIDDIR, "ceps_grid.npy"))

    trajs = []
    for i in range(len(STARTS)):
        path = os.path.join(PAPER, f"start_{i}", "trajectory.npy")
        trajs.append(np.load(path) if os.path.exists(path) else None)
        if trajs[-1] is None:
            print(f"WARNING: {path} missing, that descent will be omitted")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # The loss spans orders of magnitude, so plot it relative to the lowest sampled
    # node on a log colour scale. shading="nearest" draws one flat block per
    # evaluated node: every colour on screen is a number that was actually computed,
    # with no interpolation inventing a minimum between samples.
    rel = grid / np.nanmin(grid)
    im = axs[0].pcolormesh(ck, ceps, rel.T, cmap="viridis_r", shading="nearest",
                           norm=LogNorm(vmin=1.0, vmax=np.nanmax(rel)))
    fig.colorbar(im, ax=axs[0], label="loss / lowest sampled loss")

    for i, (traj, color) in enumerate(zip(trajs, COLORS)):
        if traj is None:
            continue
        # Path dashed and faint: the endpoints and the landscape carry the result,
        # the route between them is context.
        axs[0].plot(traj[:, 0], traj[:, 1], "--", color=color, lw=1.2, alpha=0.45, zorder=3)
        axs[0].plot(traj[0, 0], traj[0, 1], "o", color=color, ms=8, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=6)
        axs[0].plot(traj[-1, 0], traj[-1, 1], "s", color=color, ms=9, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=6)

    axs[0].plot(TRUE_C_K, TRUE_C_EPS, "*", color="gold", ms=26, markeredgecolor="black",
                markeredgewidth=1.0, zorder=4)
    axs[0].annotate("true", xy=(TRUE_C_K, TRUE_C_EPS), xytext=(10, -16),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    axs[0].set_xlabel("$c_k$")
    axs[0].set_ylabel(r"$c_\epsilon$")
    axs[0].set_title("Loss landscape and gradient descent paths\n"
                     "(circles: starts, squares: final iterates)")

    for i, (traj, color) in enumerate(zip(trajs, COLORS)):
        if traj is None:
            continue
        losses = traj[1:, 2]
        it = np.arange(losses.size)
        axs[1].plot(it, losses, "-", color=color, lw=0.8, alpha=0.25)
        axs[1].plot(it, smooth_loss(losses), "-", color=color, lw=2.0,
                    label=f"start {i}: ({STARTS[i][0]}, {STARTS[i][1]:.2f})")
    axs[1].set_yscale("log")
    # Clipped to the converged range. Start 0's raw curve dips five decades below
    # this at a single iterate (9.3e-6, iteration 151) and runs off the bottom of
    # the panel; that value is quoted in the text rather than stretching the axis
    # over eight empty decades.
    finite = np.concatenate([t[1:, 2][np.isfinite(t[1:, 2])] for t in trajs if t is not None])
    axs[1].set_ylim(max(float(np.percentile(finite, 1)) / 3, 1e-1), float(finite.max()) * 1.6)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("loss  (m$^2$)")
    axs[1].set_title(f"Loss during optimization\n(bold: rolling median over {SMOOTH_WINDOW} iterations)")
    axs[1].legend(fontsize=8.5)
    axs[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    p1 = os.path.join(PAPER, "figure1_landscape_multistart.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    fig.savefig(p1.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1} (+ .pdf)")

    print(f"\n{'start':>5s} {'from':>16s} {'final c_k':>10s} {'final c_eps':>12s} "
          f"{'best c_k':>9s} {'best c_eps':>11s} {'final loss':>12s}")
    for i, traj in enumerate(trajs):
        if traj is None:
            continue
        k = int(np.nanargmin(traj[1:, 2]))
        print(f"{i:5d} {'(%.2f, %.2f)' % STARTS[i]:>16s} {traj[-1, 0]:10.4f} {traj[-1, 1]:12.4f} "
              f"{traj[k, 0]:9.4f} {traj[k, 1]:11.4f} {traj[-1, 2]:12.4e}")


# --- Figure 2: target field, and its error before and after ------------------
#
# Written twice: a smoothed version for the paper, and an unsmoothed one drawn as
# flat cells. The smoothing is cosmetic and the raw version is what the model
# actually produced, so both are kept and the choice is left to the caption.

if not os.path.exists(os.path.join(SNAP, "field_target.npy")):
    print(f"WARNING: no snapshot fields in {SNAP} (run snapshots.py there); skipping figure 2")
else:
    lon = np.load(os.path.join(SNAP, "longitude.npy"))
    lat = np.load(os.path.join(SNAP, "latitude.npy"))
    domain = np.load(os.path.join(SNAP, "domain.npy"))
    params = np.load(os.path.join(SNAP, "snapshot_params.npy"))
    # Which fitted point the "after" panel shows. At 200 iterations start 0
    # converges, so the final iterate is the answer and needs no selection rule.
    # Its lowest-loss iterate happens to land within ~5e-7 of the true c_eps and
    # gives a blank map at a 12105x reduction -- a lucky single sample, not the
    # converged state, so it is not the default.
    WHICH = os.environ.get("REPORT21_SNAP_WHICH", "final")
    (init_ck, init_ce) = params[0]
    fit_ck, fit_ce = params[1] if WHICH == "best" else params[2]

    def masked(name):
        return np.where(domain, np.load(os.path.join(SNAP, f"field_{name}.npy")), np.nan)

    target = masked("target")
    err_init = masked("init") - target
    err_fit = masked(WHICH) - target

    def rmse(e):
        return float(np.sqrt(np.nanmean(e[~np.isnan(e)] ** 2)))

    # RMSEs from the raw fields; only the displayed maps are ever smoothed.
    r_i, r_b = rmse(err_init), rmse(err_fit)
    print(f"\nmean-MLD RMSE ({WHICH} iterate): before {r_i:.4g} m, after {r_b:.4g} m "
          f"({r_i / r_b:.1f}x reduction), {int(domain.sum())} columns")

    # 95th-percentile colour limit: a handful of extreme cells otherwise leave the
    # map almost entirely white. Both error panels share the limit so they compare,
    # and both versions of the figure share it so smoothed and raw compare too.
    vmax = float(np.nanpercentile(np.abs(err_init[~np.isnan(err_init)]), 95))
    # Robust colour limits for the target: most columns sit between -35 and -50 m
    # while a handful reach -180, which on a full-range scale renders the whole
    # field one colour.
    t_lo, t_hi = (float(np.nanpercentile(target, q)) for q in (2, 98))

    # Imported here so figure 1 still draws on an install without cartopy. Needs
    # shapely >= 2.0.5 under numpy 2: shapely 2.0.4 imports fine but every
    # projected draw dies with "ufunc 'create_collection' not supported".
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Pacific-centred: the error signal sits in the North and equatorial Pacific,
    # and the Robinson seam then falls at 20E, mostly on Africa and Europe.
    PROJ = ccrs.Robinson(central_longitude=200)
    DATA_CRS = ccrs.PlateCarree()
    FINE = 0.5  # degrees; display grid for the smoothed maps
    LAND_COLOR = "#cfcfcf"
    NODATA_COLOR = "#f2efe9"  # ocean columns outside the common domain

    def _interp_axis(a, x, x_new, axis):
        return np.apply_along_axis(lambda v: np.interp(x_new, x, v), axis, a)

    def refine(field, step=FINE):
        """Bilinear resampling of a (lon, lat) field onto a `step`-degree grid.

        Display only, and the replacement for gouraud shading, which cartopy does
        not wrap across the date line and which crops half a cell at each edge.
        Same renormalisation as smooth_map: data (NaN -> 0) and validity are
        interpolated separately and divided, so coastal values are not pulled
        toward zero, and a fine cell is kept only where more than half its weight
        is ocean. The 0.5 contour of the validity is the edge of the original
        cells, so the coast is rounded at the corners but never moved offshore.
        Periodic in longitude, clamped in latitude.
        """
        valid = np.isfinite(field).astype(float)
        data = np.where(np.isfinite(field), field, 0.0)
        lon_ext = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
        half = 0.5 * (lat[1] - lat[0])
        lon_f = np.arange(step / 2, 360.0, step)
        lat_f = np.arange(lat[0] - half + step / 2, lat[-1] + half, step)
        d, v = (
            _interp_axis(_interp_axis(np.concatenate([a[-1:], a, a[:1]]), lon_ext, lon_f, 0),
                         lat, lat_f, 1)
            for a in (data, valid)
        )
        return lon_f, lat_f, np.where(v > 0.5, d / np.maximum(v, 1e-12), np.nan)

    def figure2(sigma, suffix, note):
        def show(ax, field, cmap, **kw):
            if sigma > 0:
                x, y, f = refine(smooth_map(field, sigma))
            else:
                x, y, f = lon, lat, field
            return ax.pcolormesh(x, y, f.T, cmap=cmap, shading="auto", transform=DATA_CRS,
                                 rasterized=True, **kw)

        fig = plt.figure(figsize=(16, 4.2))
        gs = fig.add_gridspec(1, 3, wspace=0.04, left=0.02, right=0.98, top=0.84, bottom=0.2)
        ax_t, ax_b, ax_a = (fig.add_subplot(gs[0, i], projection=PROJ) for i in range(3))

        def titled(ax, title, sub):
            # Bold title over a plain subtitle; set_title cannot mix weights.
            ax.set_title(sub, fontsize=10, pad=4)
            ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 20),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=12.5, fontweight="bold")

        im_t = show(ax_t, target, "viridis", vmin=t_lo, vmax=t_hi)
        titled(ax_t, "Target mean mixed-layer depth",
               f"$c_k$ = {TRUE_C_K},  $c_\\epsilon$ = {TRUE_C_EPS}  (truth)")

        for ax, e, title, ck_, ce_, r in (
            (ax_b, err_init, "Error before assimilation", init_ck, init_ce, r_i),
            (ax_a, err_fit, "Error after assimilation", fit_ck, fit_ce, r_b),
        ):
            im_e = show(ax, e, "RdBu_r", vmin=-vmax, vmax=vmax)
            titled(ax, title, f"$c_k$ = {ck_:.4f},  $c_\\epsilon$ = {ce_:.4f}   |   "
                              f"RMSE {r:.4g} m")

        for ax in (ax_t, ax_b, ax_a):
            ax.set_global()
            ax.set_facecolor(NODATA_COLOR)
            ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR,
                           edgecolor="none", zorder=2)
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555555",
                           linewidth=0.4, zorder=3)
            ax.gridlines(color="#7f7f7f", linewidth=0.3, alpha=0.5, linestyle="--",
                         xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
            ax.spines["geo"].set_linewidth(0.6)

        # The projection fixes each map's aspect, so the maps end up smaller than
        # their grid cells; place the colour bars from the drawn positions instead.
        fig.canvas.draw()

        def cbar_below(left, right, frac, im, label):
            p0, p1 = left.get_position(), right.get_position()
            w = (p1.x1 - p0.x0) * frac
            cax = fig.add_axes([0.5 * (p0.x0 + p1.x1) - w / 2, p0.y0 - 0.075, w, 0.035])
            fig.colorbar(im, cax=cax, orientation="horizontal", extend="both").set_label(label)

        cbar_below(ax_t, ax_t, 0.7, im_t, "mixed-layer depth (m)")
        cbar_below(ax_b, ax_a, 0.45, im_e, "mixed-layer depth error, fitted - target (m)")

        path = os.path.join(PAPER, f"figure2_mld_before_after{suffix}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"saved {path} (+ .pdf) -- {note}")

    figure2(MAP_SIGMA, "", f"Gaussian smoothed, sigma = {MAP_SIGMA} cells, "
            f"resampled to {FINE} deg")
    figure2(0.0, "_raw", "unsmoothed, one flat cell per grid column")
