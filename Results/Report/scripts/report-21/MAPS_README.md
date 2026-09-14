# Redesigning the mixed-layer-depth maps

Everything the map figure needs is already on disk as `.npy`. Nothing has to be
recomputed, no cluster job is needed, and the figure rebuilds in about two seconds.

## Where the data is

All in `Results/Report/figures/report-21-paper200/start_0/`, written by
`snapshots.py` (Grid5000 job 3096899):

| file | shape | what it is |
|---|---|---|
| `field_target.npy` | (90, 40) float64 | mean MLD at the **true** parameters `c_k` = 0.1, `c_eps` = 0.7 |
| `field_init.npy` | (90, 40) | mean MLD at the initial guess, `c_k` = 0.05, `c_eps` = 0.40 |
| `field_final.npy` | (90, 40) | mean MLD at the fitted (converged) parameters |
| `field_best.npy` | (90, 40) | mean MLD at the lowest-loss iterate -- see the warning below |
| `domain.npy` | (90, 40) bool | True on the 2311 columns valid in **all four** rollouts |
| `longitude.npy` | (90,) | cell centres, 2 to 358 degrees, 4 degree spacing |
| `latitude.npy` | (40,) | cell centres, -78 to 78 degrees, 4 degree spacing |
| `snapshot_params.npy` | (4, 2) | the `(c_k, c_eps)` of each rollout, in the order init, best, final, true |
| `err_init.npy`, `err_final.npy`, `err_best.npy` | (90, 40) | the same differences against target, precomputed and already NaN-masked |

The observable is the mixed-layer depth **averaged over the day-8, 9 and 10
states** of a 10-day coupled rollout, in metres.

## The conventions that will bite you

**Axis order is (longitude, latitude).** `field.shape == (90, 40)` with longitude
first, so every `pcolormesh(lon, lat, field)` call needs `field.T`.

**Off-domain cells are 0.0, not NaN,** in the `field_*.npy` files. Mask before
doing anything: `np.where(domain, field, np.nan)`. Land really does read exactly
0.0 rather than NaN in this model, so an `isnan` test finds nothing and a plot
made without masking shows a continent-shaped block of "zero depth" that wrecks
the colour scale. The `err_*.npy` files are already masked.

**Depth is negative down.** The target runs -181 to -35 m. If you would rather
plot positive depth, negate the field *and* think about what that does to the
error panels' sign, which share the convention.

**Longitude is periodic**, 2..358 at cell centres, on the real-topography
`global_4deg_learning` grid: the cells line up with Natural Earth coastlines with
longitude 0 at Greenwich, so no offset is needed when projecting. A map drawn
from cell centres without wrapping shows a one-cell seam at the date line;
`pcolormesh(..., shading="auto")` with `transform=PlateCarree()` wraps correctly.

**Off-domain is not the same as land.** The domain also drops ocean columns
whose MLD is not well defined in every rollout -- most of the Arctic, the
Mediterranean, Hudson Bay, parts of the Weddell and Ross seas. The maps
therefore draw real land in grey from Natural Earth and paint off-domain ocean a
separate pale colour; filling everything off-domain as land would redraw the
coastline wrongly.

**The domain is the intersection of all four rollouts**, so every panel covers
identical columns and the RMSEs are comparable. Don't recompute a per-panel mask.

## The existing script

`Results/Report/scripts/report-21/paper_figures.py` -- reads only `.npy`, draws
both paper figures, run it with no arguments from the repo root:

    python3 Results/Report/scripts/report-21/paper_figures.py

The map figure is the `figure2(...)` function near the bottom. It is called twice
and writes, into `figures/report-21-paper200/`:

    figure2_mld_before_after.png/.pdf       Gaussian smoothed, resampled to 0.5 deg
    figure2_mld_before_after_raw.png/.pdf   unsmoothed, one flat 4-degree cell per column

Both are cartopy maps: Robinson projection centred on 200E (the signal is in the
Pacific, and the seam then falls at 20E, mostly on Africa), Natural Earth 110m
land and coastline drawn over the data, colour bars below.

**Cartopy needs shapely >= 2.0.5 under numpy 2.** The `diffusion` env has
numpy 2.5.2 with shapely 2.0.4: `import cartopy` works, but every projected draw
dies with `TypeError: ufunc 'create_collection' not supported`. Either run with
the `veros` env (numpy 1.26, works as is) or `pip install "shapely>=2.1"` in
`diffusion` (tested: 2.1.2 fixes it). Figure 1 does not import cartopy.

Environment variables:

| variable | default | effect |
|---|---|---|
| `REPORT21_MAP_SIGMA` | `1.0` | Gaussian width in grid cells; `0` disables smoothing |
| `REPORT21_SNAP_WHICH` | `final` | which fitted point the "after" panel shows; `best` for the lowest-loss iterate |
| `REPORT21_SNAP_DIR` | start_0 of the current set | read fields from somewhere else |

Three helpers worth reusing rather than rewriting (`refine` is defined inside
the figure-2 block, next to `figure2`):

- `smooth_map(field, sigma)` -- NaN-aware Gaussian smoothing. Data and validity
  mask are convolved separately and divided, so coastal cells average only over
  real ocean neighbours instead of being pulled toward zero by the land; periodic
  in longitude, edge-clamped in latitude; re-masked at the end so the field cannot
  grow into land.
- `smooth_loss(y)` -- rolling median in log space, for the other figure.
- `refine(field)` -- bilinear resampling onto a 0.5 degree display grid, with
  the same data/validity renormalisation, keeping a fine cell only where more
  than half its weight is ocean. This is what removes the blocky look; the
  Gaussian alone just produces soft-edged squares. It replaced
  `shading="gouraud"`, which cartopy does not wrap across the date line and
  which cropped half a cell at each edge. The 0.5 validity contour sits on the
  original cell edges, so coasts are rounded at the corners but never moved.

## Things not to change without saying so in the caption

**Every RMSE is computed from unsmoothed fields.** Smoothing is display-only. If
you compute an RMSE from a smoothed map it will come out lower and will not match
the report.

**Both error panels share one colour limit** -- the 95th percentile of the
*before* error, currently about 0.28 m -- and so do the smoothed and raw versions,
so all four panels compare. Using a per-panel limit makes the "after" panel look
as bad as the "before" one.

**The target panel uses 2nd-98th percentile limits.** A handful of columns reach
-180 m while most sit between -35 and -50 m; on a full-range scale the whole field
renders as one flat colour.

The raw and smoothed versions disagree about amplitude for a real reason: the
error is cell-scale and dipolar, with adjacent columns saturating opposite ends of
the colourbar in the 20-40N band and along the equator. Smoothing averages those
neighbours toward zero, so the smoothed map reads calmer than an RMSE of 0.7691 m
suggests. The smoothed version is the readable one for spatial pattern; the raw
one is the honest one about pointwise amplitude.

## Reference numbers

Mean-MLD RMSE against the target, over the 2311 columns:

| | RMSE |
|---|---|
| before, `c_k` = 0.05, `c_eps` = 0.40 | 0.7691 m |
| after, `c_k` = 0.0999, `c_eps` = 0.6997 (converged) | 0.03976 m (19.3x) |
| after, `c_k` = 0.1000, `c_eps` = 0.7000 (lowest-loss iterate) | 0.0001 m (12105x) |

Use the converged one. The lowest-loss iterate landed within about 5e-7 of the
true `c_eps` -- a single lucky sample rather than the converged state -- and gives
a blank "after" map at a five-figure reduction factor, which is the kind of result
a reader assumes is a bug. Background in
[report-22](../../report-22-landscape-and-multistart.md).

## If the fields ever have to be regenerated

On Grid5000 (see `job_snapshots200.sh`), roughly 10 minutes: four 10-day coupled
rollouts at truth, at the initial guess, and at the two fitted points. It reads
the fitted parameters from `trajectory.npy` in the same directory, so pointing
`REPORT21_OUT_DIR` at a different run's directory snapshots that run instead.
