"""Loss landscape with the descents on it, and their loss curves.

    python figures/optimization.py --grid grid/ --run runs/start_0 --run runs/start_3 \
        --out figures/optimization.png

Left   the scan as loss / lowest sampled node, log scale, with each descent's path,
       start (circle) and final iterate (square).
Right  each run's loss against iteration, raw and smoothed.
"""

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def smoothed(losses, window):
    """Rolling median of the loss in log space, then a short moving average.

    Median, not mean: a single deep iterate would otherwise drag the curve to an
    excursion no other iterate shares. Log space because the loss spans decades.
    Windows shrink at the edges, so the curve starts and ends on real data.
    """
    log_losses = np.log10(np.maximum(np.asarray(losses, float), np.finfo(float).tiny))
    half = window // 2
    median = np.array([np.median(log_losses[max(0, i - half):i + half + 1])
                       for i in range(log_losses.size)])
    return 10.0 ** np.array([median[max(0, i - 2):i + 3].mean() for i in range(median.size)])


@click.command()
@click.option("--grid", type=click.Path(path_type=Path, exists=True), required=True,
              help="Directory holding loss_grid.npy and the axis values.")
@click.option("--run", "runs", type=click.Path(path_type=Path, exists=True), multiple=True,
              required=True, help="Calibration directory; repeat once per descent.")
@click.option("--true-c-k", default=0.1, show_default=True)
@click.option("--true-c-eps", default=0.7, show_default=True)
@click.option("--loss-units", default="variance-normalised", show_default=True)
@click.option("--smooth-window", default=9, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(grid, runs, true_c_k, true_c_eps, loss_units, smooth_window, out):
    losses = np.load(grid / "loss_grid.npy")
    c_k_values = np.load(grid / "c_k_values.npy")
    c_eps_values = np.load(grid / "c_eps_values.npy")
    trajectories = [np.load(run / "trajectory.npy") for run in runs]

    figure, (landscape, curves) = plt.subplots(1, 2, figsize=(14, 5.6),
                                               gridspec_kw={"width_ratios": [1.15, 1]})
    plt.rcParams.update({"font.size": 11})

    # shading="nearest" draws one flat block per evaluated node, so every colour on
    # screen is a number that was computed rather than an interpolation between two.
    relative = losses / np.nanmin(losses)
    mesh = landscape.pcolormesh(c_k_values, c_eps_values, relative.T, cmap="viridis_r",
                                shading="nearest", norm=LogNorm(vmin=1.0, vmax=np.nanmax(relative)))
    figure.colorbar(mesh, ax=landscape, label="loss / lowest sampled loss")

    for trajectory, color in zip(trajectories, COLORS):
        landscape.plot(trajectory[:, 0], trajectory[:, 1], "--", color=color, lw=1.2, alpha=0.45)
        landscape.plot(trajectory[0, 0], trajectory[0, 1], "o", color=color, ms=8,
                       markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        landscape.plot(trajectory[-1, 0], trajectory[-1, 1], "s", color=color, ms=9,
                       markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    landscape.plot(true_c_k, true_c_eps, "*", color="gold", ms=26, markeredgecolor="black", zorder=4)
    landscape.annotate("true", xy=(true_c_k, true_c_eps), xytext=(10, -16),
                       textcoords="offset points", fontsize=10, fontweight="bold")
    landscape.set_xlabel("$c_k$")
    landscape.set_ylabel(r"$c_\epsilon$")
    landscape.set_title("Loss landscape and gradient descent paths\n"
                        "(circles: starts, squares: final iterates)")

    for run, trajectory, color in zip(runs, trajectories, COLORS):
        values = trajectory[1:, 2]
        iterations = np.arange(values.size)
        curves.plot(iterations, values, "-", color=color, lw=0.8, alpha=0.25)
        curves.plot(iterations, smoothed(values, smooth_window), "-", color=color, lw=2.0,
                    label=f"{run.name}: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})")
    curves.set_yscale("log")
    curves.set_xlabel("iteration")
    curves.set_ylabel(f"loss ({loss_units})")
    curves.set_title(f"Loss during optimization\n(bold: rolling median over {smooth_window} iterations)")
    curves.legend(fontsize=8.5)
    curves.grid(alpha=0.3, which="both")

    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    click.echo(f"saved {out} (+ .pdf)")

    click.echo(f"\n{'run':>16s} {'final c_k':>10s} {'final c_eps':>12s} {'final loss':>12s}")
    for run, trajectory in zip(runs, trajectories):
        click.echo(f"{run.name:>16s} {trajectory[-1, 0]:10.4f} {trajectory[-1, 1]:12.4f} "
                   f"{trajectory[-1, 2]:12.4e}")


if __name__ == "__main__":
    main()
