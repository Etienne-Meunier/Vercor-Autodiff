"""Fit (c_k, c_eps) to a synthetic target by gradient descent through the coupled model.

    python calibrate.py --observable tsdiff_avg --c-k 0.15 --c-eps 0.40 --out runs/start_3

Writes trajectory.npy, one row per iteration:
(c_k, c_eps, loss, dL/dc_k, dL/dc_eps). Row 0 holds the start point; row k+1 holds
the parameters produced after iteration k alongside the loss evaluated before it.
"""

import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import CoupledModel
from observables import OBSERVABLES, make_loss


@click.command()
@click.option("--observable", type=click.Choice(sorted(OBSERVABLES)), default="tsdiff_avg",
              show_default=True)
@click.option("--days", default=10, show_default=True, help="Rollout length in days.")
@click.option("--restore-to-climatology/--no-restore-to-climatology", default=True,
              show_default=True,
              help="Relax sea surface temperature toward observed climatology.")
@click.option("--average-days", default=3, show_default=True,
              help="Average the observable over this many final daily states.")
@click.option("--c-k", default=0.05, show_default=True, help="Starting c_k.")
@click.option("--c-eps", default=0.40, show_default=True, help="Starting c_eps.")
@click.option("--true-c-k", default=0.1, show_default=True, help="c_k the target is generated at.")
@click.option("--true-c-eps", default=0.7, show_default=True, help="c_eps the target is generated at.")
@click.option("--iterations", default=200, show_default=True)
@click.option("--learning-rate", default=2e-2, show_default=True,
              help="Peak value of the cosine-decay schedule.")
@click.option("--clip-factor", default=1.2, show_default=True,
              help="Clip the gradient norm at this multiple of its start-point value.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output directory.")
def main(observable, days, average_days, restore_to_climatology, c_k, c_eps, true_c_k, true_c_eps,
         iterations, learning_rate, clip_factor, out):
    out.mkdir(parents=True, exist_ok=True)
    obs = OBSERVABLES[observable]
    click.echo(f"observable={obs.name}, {days}-day rollout, mean over the last {average_days} days, "
               f"{iterations} iterations, loss in {obs.units}")

    model = CoupledModel(days=days, average_days=average_days,
                          restore_to_climatology=restore_to_climatology)

    started = time.time()
    loss_terms, target, valid, weights = make_loss(model, obs, true_c_k, true_c_eps)
    click.echo(f"target generated at c_k={true_c_k}, c_eps={true_c_eps} "
               f"({time.time() - started:.1f}s incl. compile); "
               f"{int(np.asarray(valid).sum())} cells in the loss")
    click.echo(f"loss weights: {np.asarray(weights)}")

    @jax.jit
    def value_and_grad(params):
        return jax.value_and_grad(lambda p: jnp.sum(loss_terms(p["c_k"], p["c_eps"])))(params)

    params = {"c_k": jnp.asarray(c_k), "c_eps": jnp.asarray(c_eps)}

    # The clip threshold is measured at the start point rather than fixed, so that one
    # rule transfers between observables whose losses carry different units.
    loss, grad = value_and_grad(params)
    start_norm = float(np.hypot(float(grad["c_k"]), float(grad["c_eps"])))
    clip = clip_factor * start_norm
    click.echo(f"start loss {float(loss):.5e}, |g| {start_norm:.4e}, "
               f"gradient norm clipped at {clip:.4e}")

    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=iterations, alpha=0.0)
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optax.adam(schedule))
    opt_state = optimizer.init(params)

    trajectory = [(c_k, c_eps, np.nan, np.nan, np.nan)]
    click.echo(f"\n{'iter':>4s} {'loss':>13s} {'c_k':>9s} {'c_eps':>9s} "
               f"{'dL/dc_k':>12s} {'dL/dc_eps':>12s} {'s':>6s}")
    for iteration in range(iterations):
        started = time.time()
        if iteration > 0:
            loss, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        trajectory.append((float(params["c_k"]), float(params["c_eps"]), float(loss),
                           float(grad["c_k"]), float(grad["c_eps"])))
        np.save(out / "trajectory.npy", np.array(trajectory, float))
        click.echo(f"{iteration:4d} {float(loss):13.5e} {float(params['c_k']):9.4f} "
                   f"{float(params['c_eps']):9.4f} {float(grad['c_k']):12.3e} "
                   f"{float(grad['c_eps']):12.3e} {time.time() - started:6.0f}")

    history = np.array(trajectory, float)
    best = int(np.nanargmin(history[1:, 2]))
    click.echo(f"\nfinal  c_k={history[-1, 0]:.4f} ({abs(history[-1, 0] - true_c_k) / true_c_k:.1%} err)  "
               f"c_eps={history[-1, 1]:.4f} ({abs(history[-1, 1] - true_c_eps) / true_c_eps:.1%} err)  "
               f"loss={history[-1, 2]:.5e}")
    click.echo(f"best   c_k={history[best, 0]:.4f}  c_eps={history[best, 1]:.4f}  "
               f"loss={history[best + 1, 2]:.5e} (iteration {best})")
    click.echo(f"saved {out / 'trajectory.npy'}")


if __name__ == "__main__":
    main()
