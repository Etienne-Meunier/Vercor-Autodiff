"""Fields for the snapshot figure: what the loss sees, and where it is wrong.

Re-runs the rollout at four parameter settings -- the truth, the start point, the
lowest-loss iterate and the final iterate of a finished calibration -- and saves for
each the observable, the top-level potential-density difference and the loss of each
column.

    python snapshot.py --run runs/start_3 --c-k 0.15 --c-eps 0.40
"""

import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np

from model import CoupledModel
from observables import OBSERVABLES, make_loss


@click.command()
@click.option("--run", type=click.Path(path_type=Path, exists=True), required=True,
              help="Calibration directory holding trajectory.npy; fields are written beside it.")
@click.option("--observable", type=click.Choice(sorted(OBSERVABLES)), default="tsdiff_avg",
              show_default=True)
@click.option("--days", default=10, show_default=True)
@click.option("--restore-to-climatology/--no-restore-to-climatology", default=True,
              show_default=True,
              help="Relax sea surface temperature toward observed climatology.")
@click.option("--average-days", default=3, show_default=True)
@click.option("--c-k", default=0.05, show_default=True, help="Start point of that calibration.")
@click.option("--c-eps", default=0.40, show_default=True, help="Start point of that calibration.")
@click.option("--true-c-k", default=0.1, show_default=True)
@click.option("--true-c-eps", default=0.7, show_default=True)
def main(run, observable, days, average_days, restore_to_climatology, c_k, c_eps, true_c_k, true_c_eps):
    trajectory = np.load(run / "trajectory.npy")
    # Row k+1 holds the parameters produced after iteration k and the loss evaluated
    # before it, so the lowest-loss iterate's parameters are read from row k.
    best = int(np.nanargmin(trajectory[1:, 2]))
    points = {
        "target": (true_c_k, true_c_eps),
        "start": (c_k, c_eps),
        "best": (float(trajectory[best, 0]), float(trajectory[best, 1])),
        "final": (float(trajectory[-1, 0]), float(trajectory[-1, 1])),
    }
    click.echo(f"{trajectory.shape[0] - 1} iterations; "
               f"best loss {np.nanmin(trajectory[1:, 2]):.5e} at iteration {best}")
    for name, (a, b) in points.items():
        click.echo(f"  {name:>6s}: c_k={a:.4f}, c_eps={b:.4f}")

    obs = OBSERVABLES[observable]
    model = CoupledModel(days=days, average_days=average_days,
                          restore_to_climatology=restore_to_climatology)
    loss_terms, target, valid, weights = make_loss(model, obs, true_c_k, true_c_eps)
    surface = model.grid.surface

    @jax.jit
    def diagnostics(a, b):
        """The observable, the per-column loss and the top-level density difference."""
        states = model.ocean_states(a, b)[-average_days:]
        field, cell_valid = obs.average(states, model.grid)
        squared = weights * jnp.where(valid & cell_valid, field - target, 0.0) ** 2
        column_loss = jnp.sum(squared, axis=-1) if squared.ndim == 3 else squared
        density_difference = sum(
            state.variables.prho[2:-2, 2:-2, surface - 1] - state.variables.prho[2:-2, 2:-2, surface]
            for state in states) / len(states)
        return field, column_loss, density_difference

    for name, (a, b) in points.items():
        started = time.time()
        field, column_loss, density_difference = diagnostics(jnp.asarray(a), jnp.asarray(b))
        np.save(run / f"field_{name}.npy", np.asarray(field))
        np.save(run / f"column_loss_{name}.npy", np.asarray(column_loss))
        np.save(run / f"density_difference_{name}.npy", np.asarray(density_difference))
        click.echo(f"  {name:>6s}: total loss {float(jnp.sum(column_loss)):.6e} "
                   f"({time.time() - started:.1f}s)")

    np.save(run / "column_mask.npy", model.grid.mask[:, :, surface - 1])
    np.save(run / "longitude.npy", model.grid.longitude)
    np.save(run / "latitude.npy", model.grid.latitude)
    np.save(run / "snapshot_points.npy", np.array(list(points.values()), float))
    click.echo(f"saved fields in {run}")


if __name__ == "__main__":
    main()
