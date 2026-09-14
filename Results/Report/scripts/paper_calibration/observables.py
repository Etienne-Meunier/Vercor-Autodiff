"""What the calibration compares: two observables and the loss built from them.

mld_avg     mixed-layer depth, averaged over the last days of the rollout
tsdiff_avg  temperature and salinity differences over the top three levels,
            averaged the same way

An observable maps ocean states to (field, valid) and carries the weights its
squared error is summed with. The loss is always

    sum over cells and channels of  weight * (field - target)^2

evaluated on the target's valid cells, so the loss domain never moves between
iterations.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from model import CoupledModel, Grid

MLD_REFERENCE_DEPTH = -10.0    # m, negative down
MLD_REFERENCE_OFFSET = 0.03    # kg/m^3 density excess defining the mixed-layer base


def _bracketing_levels(prho, mask, zt):
    """Levels bracketing the mixed-layer base. Discrete, so kept off the gradient path."""
    level = jnp.arange(zt.shape[-1])
    reference = jnp.max(jnp.where(zt < MLD_REFERENCE_DEPTH, level, -1))

    threshold = prho[:, :, reference] + MLD_REFERENCE_OFFSET
    usable = mask & (level <= reference)
    excess = prho - threshold[:, :, jnp.newaxis]

    below = usable & (excess > 0)
    index_below = jnp.argmax(jnp.where(below, zt, -jnp.inf), axis=-1)
    above = usable & (excess < 0) & (zt > zt[index_below][:, :, jnp.newaxis])
    index_above = jnp.argmin(jnp.where(above, zt, jnp.inf), axis=-1)

    defined = jnp.any(below, axis=-1) & jnp.any(above, axis=-1)
    return reference, index_below, index_above, defined


def mixed_layer_depth(payload, grid: Grid):
    """Depth where potential density first exceeds its reference value, interpolated.

    The division is guarded before it happens: a degenerate column would otherwise
    put a NaN into the gradient of the summed loss.
    """
    prho = payload.variables.prho[2:-2, 2:-2, :]
    zt, mask = jnp.asarray(grid.zt), jnp.asarray(grid.mask)
    reference, index_below, index_above, defined = _bracketing_levels(prho, mask, zt)

    threshold = prho[:, :, reference] + MLD_REFERENCE_OFFSET
    prho_below = jnp.take_along_axis(prho, index_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    prho_above = jnp.take_along_axis(prho, index_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    depth_below, depth_above = zt[index_below], zt[index_above]

    denominator = jnp.where(defined, prho_above - prho_below, 1.0)
    depth = (threshold - prho_below) / denominator * (depth_above - depth_below) + depth_below
    return jnp.where(defined, depth, 0.0), defined


def top_level_differences(payload, grid: Grid):
    """T and S differences to the surface level over the top three levels.

    Channels: T[-2]-T[-1], T[-3]-T[-1], S[-2]-S[-1], S[-3]-S[-1]. Differences rather
    than level values: the closure acts on vertical structure, while level values are
    dominated by the surface-flux common mode.
    """
    variables = payload.variables
    temp = variables.temp[2:-2, 2:-2, -3:, variables.tau]
    salt = variables.salt[2:-2, 2:-2, -3:, variables.tau]
    field = jnp.stack([temp[..., 1] - temp[..., 2], temp[..., 0] - temp[..., 2],
                       salt[..., 1] - salt[..., 2], salt[..., 0] - salt[..., 2]], axis=-1)

    surface = grid.surface
    level_2, level_3 = grid.mask[:, :, surface - 1], grid.mask[:, :, surface - 2]
    valid = jnp.asarray(np.stack([level_2, level_3, level_2, level_3], axis=-1))
    return field, valid


@dataclass(frozen=True)
class Observable:
    name: str
    units: str
    of_state: Callable      # (payload, grid) -> (field, valid)
    intersect_valid: bool   # True when a cell can lose validity as parameters move

    def average(self, states, grid: Grid):
        """Mean of the observable over the given states, with their common valid cells."""
        pairs = [self.of_state(state, grid) for state in states]
        valid = pairs[0][1]
        if self.intersect_valid:
            for _, cell_valid in pairs[1:]:
                valid = valid & cell_valid
        field = sum(field for field, _ in pairs) / len(pairs)
        return jnp.where(valid, field, 0.0), valid

    def weights(self, target, valid):
        """Per-channel weights of the squared error.

        The MLD is a single field in m^2 and needs none. The four T/S channels carry
        different units and variances, so each is normalised by the cell count and
        spatial variance of its own target channel, making the four terms comparable.
        Weights depend on the target alone, so every evaluation shares one loss.
        """
        if self.name != "tsdiff_avg":
            return 1.0
        target, valid = np.asarray(target), np.asarray(valid)
        return jnp.asarray([1.0 / (valid[..., c].sum() * target[..., c][valid[..., c]].var())
                            for c in range(target.shape[-1])])


OBSERVABLES = {
    "mld_avg": Observable("mld_avg", "m^2", mixed_layer_depth, intersect_valid=True),
    "tsdiff_avg": Observable("tsdiff_avg", "variance-normalised",
                             top_level_differences, intersect_valid=False),
}


def make_loss(model: CoupledModel, observable: Observable, true_c_k: float, true_c_eps: float):
    """Build the loss, its target field and its per-channel weights.

    Returns (loss_terms, target, valid, weights), where loss_terms(c_k, c_eps) gives
    one loss per observable channel; their sum is the loss that is descended.
    """
    def field(c_k, c_eps):
        return observable.average(model.ocean_states(c_k, c_eps)[-model.average_days:], model.grid)

    target, target_valid = field(jnp.asarray(true_c_k), jnp.asarray(true_c_eps))
    target = jnp.asarray(np.asarray(target))          # frozen: never differentiated through
    target_valid = jnp.asarray(np.asarray(target_valid))
    weights = observable.weights(target, target_valid)

    def loss_terms(c_k, c_eps):
        value, valid = field(c_k, c_eps)
        squared = weights * jnp.where(target_valid & valid, value - target, 0.0) ** 2
        return jnp.sum(squared, axis=(0, 1)) if squared.ndim == 3 else jnp.sum(squared)[None]

    return loss_terms, target, target_valid, weights
