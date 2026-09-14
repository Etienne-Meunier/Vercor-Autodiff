"""Forward-only scan of the loss over a (c_k, c_eps) grid, and the merge of its chunks.

The scan is split by c_k column so several jobs can share it, and it checkpoints after
every node, so an interrupted job resumes where it stopped:

    python landscape.py scan --col-start 0 --col-end 4 --out grid/
    python landscape.py merge --out grid/

`loss_grid[i, j]` is indexed by (c_k, c_eps). Each node's per-channel losses are kept
alongside the total, so a single term's landscape can be drawn afterwards.
"""

import re
import time
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np

from model import CoupledModel
from observables import OBSERVABLES, make_loss


@click.group()
def main():
    pass


@main.command()
@click.option("--observable", type=click.Choice(sorted(OBSERVABLES)), default="tsdiff_avg",
              show_default=True)
@click.option("--days", default=10, show_default=True)
@click.option("--restore-to-climatology/--no-restore-to-climatology", default=True,
              show_default=True,
              help="Relax sea surface temperature toward observed climatology.")
@click.option("--average-days", default=3, show_default=True)
@click.option("--true-c-k", default=0.1, show_default=True)
@click.option("--true-c-eps", default=0.7, show_default=True)
@click.option("--grid-size", default=16, show_default=True, help="Nodes per axis.")
@click.option("--c-k-range", nargs=2, type=float, default=(0.02, 0.18), show_default=True)
@click.option("--c-eps-range", nargs=2, type=float, default=(0.25, 1.05), show_default=True)
@click.option("--col-start", default=0, show_default=True, help="First c_k column of this chunk.")
@click.option("--col-end", default=None, type=int, help="One past the last c_k column [default: all].")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def scan(observable, days, average_days, restore_to_climatology, true_c_k, true_c_eps, grid_size,
         c_k_range, c_eps_range, col_start, col_end, out):
    """Evaluate the loss on a grid of parameters (no gradients)."""
    out.mkdir(parents=True, exist_ok=True)
    col_end = grid_size if col_end is None else col_end
    c_k_values = np.linspace(*c_k_range, grid_size)
    c_eps_values = np.linspace(*c_eps_range, grid_size)
    np.save(out / "c_k_values.npy", c_k_values)
    np.save(out / "c_eps_values.npy", c_eps_values)

    obs = OBSERVABLES[observable]
    click.echo(f"observable={obs.name}, {grid_size}x{grid_size} grid, "
               f"c_k {c_k_range[0]}..{c_k_range[1]}, c_eps {c_eps_range[0]}..{c_eps_range[1]}, "
               f"columns [{col_start}, {col_end})")

    model = CoupledModel(days=days, average_days=average_days,
                          restore_to_climatology=restore_to_climatology)
    loss_terms, _, valid, weights = make_loss(model, obs, true_c_k, true_c_eps)
    loss_terms = jax.jit(loss_terms)
    click.echo(f"{int(np.asarray(valid).sum())} cells in the loss; weights {np.asarray(weights)}")

    total_path = out / f"loss_chunk_{col_start:02d}_{col_end:02d}.npy"
    terms_path = out / f"loss_terms_chunk_{col_start:02d}_{col_end:02d}.npy"
    n_terms = int(np.asarray(loss_terms(jnp.asarray(true_c_k), jnp.asarray(true_c_eps))).size)
    total = np.load(total_path) if total_path.exists() else np.full((col_end - col_start, grid_size), np.nan)
    terms = np.load(terms_path) if terms_path.exists() else np.full(total.shape + (n_terms,), np.nan)
    if total_path.exists():
        click.echo(f"resuming: {int(np.isfinite(total).sum())}/{total.size} nodes already done")

    for i in range(col_start, col_end):
        for j, c_eps in enumerate(c_eps_values):
            if np.isfinite(total[i - col_start, j]):
                continue
            started = time.time()
            node = np.asarray(loss_terms(jnp.asarray(c_k_values[i]), jnp.asarray(c_eps)))
            terms[i - col_start, j] = node
            total[i - col_start, j] = node.sum()
            np.save(terms_path, terms)
            np.save(total_path, total)
            click.echo(f"  ({i:2d},{j:2d}) c_k={c_k_values[i]:.4f} c_eps={c_eps:.4f}  "
                       f"loss={node.sum():.6e}  ({time.time() - started:.1f}s)")
    click.echo(f"chunk minimum {np.nanmin(total):.6e}, maximum {np.nanmax(total):.6e}")


@main.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
def merge(out):
    """Assemble the chunks into loss_grid.npy, reporting any node still missing."""
    c_k_values = np.load(out / "c_k_values.npy")
    c_eps_values = np.load(out / "c_eps_values.npy")

    for prefix, name in (("loss_chunk", "loss_grid.npy"), ("loss_terms_chunk", "loss_terms_grid.npy")):
        grid = None
        for path in sorted(out.glob(f"{prefix}_*.npy")):
            first, last = (int(g) for g in re.match(rf"{prefix}_(\d+)_(\d+)", path.stem).groups())
            chunk = np.load(path)
            if grid is None:
                grid = np.full((c_k_values.size, c_eps_values.size) + chunk.shape[2:], np.nan)
            grid[first:last] = chunk
            click.echo(f"{path.name}: columns [{first}, {last}), "
                       f"{int(np.isfinite(chunk).sum())}/{chunk.size} nodes")
        if grid is None:
            continue
        missing = int(np.isnan(grid[..., 0] if grid.ndim == 3 else grid).sum())
        click.echo(f"merged {name} {grid.shape}: {missing} missing nodes")
        np.save(out / name, grid)

    grid = np.load(out / "loss_grid.npy")
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
    click.echo(f"lowest node: c_k={c_k_values[i]:.4f}, c_eps={c_eps_values[j]:.4f}, "
               f"loss={grid[i, j]:.6e}")


if __name__ == "__main__":
    main()
