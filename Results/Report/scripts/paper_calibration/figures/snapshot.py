"""What the loss measures, and where it is wrong.

    python figures/snapshot.py --run runs/start_3 --out figures/snapshot.png

Left    the target's top-level potential-density difference, rho[-2] - rho[-1],
        the quantity both the MLD and the T/S channels respond to
Middle  each column's loss before calibration
Right   the same after calibration

The loss panels share a logarithmic colour scale; on a linear one the "after" panel
is blank. Needs cartopy (shapely >= 2.0.5 under numpy 2).
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROJECTION = ccrs.Robinson(central_longitude=200)  # Pacific-centred: the seam falls on Africa
DATA_CRS = ccrs.PlateCarree()
LAND_COLOR, NODATA_COLOR = "#cfcfcf", "#f2efe9"


@click.command()
@click.option("--run", type=click.Path(path_type=Path, exists=True), required=True,
              help="Calibration directory that snapshot.py has written fields into.")
@click.option("--iterate", type=click.Choice(["final", "best"]), default="final", show_default=True,
              help="Which fitted point the 'after' panel shows.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(run, iterate, out):
    longitude = np.load(run / "longitude.npy")
    latitude = np.load(run / "latitude.npy")
    mask = np.load(run / "column_mask.npy")
    (true_c_k, true_c_eps), (start_c_k, start_c_eps), best, final = np.load(run / "snapshot_points.npy")
    fitted_c_k, fitted_c_eps = final if iterate == "final" else best

    def field(name):
        return np.where(mask, np.load(run / f"{name}.npy"), np.nan)

    density_difference = field("density_difference_target")
    before, after = field("column_loss_start"), field(f"column_loss_{iterate}")
    total_before, total_after = np.nansum(before), np.nansum(after)
    click.echo(f"{int(mask.sum())} columns; loss before {total_before:.4e}, after {total_after:.4e} "
               f"({total_before / total_after:.1f}x reduction)")
    for label, values in (("before", before), ("after", after)):
        top = np.nanpercentile(values, 99)
        click.echo(f"  {label}: top 1% of columns carry "
                   f"{np.nansum(values[values >= top]) / np.nansum(values):.0%} of the loss")

    # Floor the colour scale at 1e-4 of the "before" 99.5th percentile, so that
    # near-zero columns do not stretch it over fifteen decades.
    high = float(np.nanpercentile(before, 99.5))
    low = high * 1e-4
    density_low, density_high = (float(np.nanpercentile(density_difference, q)) for q in (2, 98))

    plt.rcParams.update({"font.size": 11})
    figure = plt.figure(figsize=(16, 4.2))
    spec = figure.add_gridspec(1, 3, wspace=0.04, left=0.02, right=0.98, top=0.84, bottom=0.2)
    axes = [figure.add_subplot(spec[0, i], projection=PROJECTION) for i in range(3)]

    def titled(axis, title, subtitle):
        axis.set_title(subtitle, fontsize=10, pad=4)  # set_title cannot mix font weights
        axis.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 20),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=12.5, fontweight="bold")

    density_mesh = axes[0].pcolormesh(longitude, latitude, density_difference.T, cmap="viridis",
                                      shading="auto", transform=DATA_CRS, rasterized=True,
                                      vmin=density_low, vmax=density_high)
    titled(axes[0], "Target top-level density difference",
           f"$c_k$ = {true_c_k},  $c_\\epsilon$ = {true_c_eps}  (truth)")

    for axis, values, title, c_k, c_eps, total in (
        (axes[1], before, "Loss per column before", start_c_k, start_c_eps, total_before),
        (axes[2], after, "Loss per column after", fitted_c_k, fitted_c_eps, total_after),
    ):
        loss_mesh = axis.pcolormesh(longitude, latitude, np.maximum(values, low).T, cmap="magma_r",
                                    shading="auto", transform=DATA_CRS, rasterized=True,
                                    norm=LogNorm(vmin=low, vmax=high))
        titled(axis, title, f"$c_k$ = {c_k:.4f},  $c_\\epsilon$ = {c_eps:.4f}   |   total {total:.2e}")

    for axis in axes:
        axis.set_global()
        axis.set_facecolor(NODATA_COLOR)
        axis.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR, edgecolor="none", zorder=2)
        axis.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555555", linewidth=0.4, zorder=3)
        axis.gridlines(color="#7f7f7f", linewidth=0.3, alpha=0.5, linestyle="--",
                       xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
        axis.spines["geo"].set_linewidth(0.6)

    # The projection fixes each map's aspect, so the axes end up smaller than their
    # grid cells; place the colour bars from the drawn positions instead.
    figure.canvas.draw()

    def colorbar(left, right, fraction, mesh, label, extend):
        left_box, right_box = left.get_position(), right.get_position()
        width = (right_box.x1 - left_box.x0) * fraction
        cax = figure.add_axes([0.5 * (left_box.x0 + right_box.x1) - width / 2,
                               left_box.y0 - 0.075, width, 0.035])
        figure.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend).set_label(label)

    colorbar(axes[0], axes[0], 0.7, density_mesh,
             r"$\rho_\theta[-2] - \rho_\theta[-1]$ (kg m$^{-3}$)", "both")
    colorbar(axes[1], axes[2], 0.45, loss_mesh, "loss contribution per column", "min")

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    click.echo(f"saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
