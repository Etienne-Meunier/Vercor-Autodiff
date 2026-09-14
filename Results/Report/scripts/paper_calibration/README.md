# Calibrating TKE closure parameters through a coupled climate model

Gradient-based calibration of the two Veros TKE closure parameters, `c_k` and `c_eps`,
by differentiating a coupled ocean-atmosphere-land rollout end to end with JAX. The
target is synthetic: the same model at known parameters, so the answer is known and
the recovery can be checked.

This directory reproduces the results of reports 22 (mixed-layer-depth observable) and
23 (top-level temperature/salinity differences).

## Layout

| file | role |
|---|---|
| `model.py` | the coupled model: Veros ocean + JCM land + JAX-GCM atmosphere on the global 4 degree grid, as a differentiable function of `(c_k, c_eps)` |
| `observables.py` | what is compared: the averaged mixed-layer depth, the averaged top-level T/S differences, and the loss built from either |
| `calibrate.py` | one gradient descent from one start point |
| `landscape.py` | forward-only scan of the loss over a parameter grid, and the merge of its chunks |
| `snapshot.py` | fields for the snapshot figure, from a finished calibration |
| `figures/optimization.py` | landscape with the descent paths, and the loss curves |
| `figures/snapshot.py` | target density difference, and the loss per column before and after |

Requirements: `vercor` with its `jcm` extra, `jax`, `optax`, `click`, `numpy`,
`matplotlib`, and `cartopy` for the snapshot figure. If `vercor` is not installed,
`model.py` falls back to the checkout at the root of this repository. Everything runs on CPU
(`JAX_PLATFORMS=cpu`); a 10-day rollout takes about 25 s and a gradient about 130 s.

## Method

The model is a chain of daily coupling steps. Both parameters enter Veros's TKE
closure, which sets the vertical mixing that shapes the upper-ocean density profile.

```
loss(c_k, c_eps):
    states <- run the coupled model for `days` days at (c_k, c_eps)
    field  <- mean over the last `average_days` states of observable(state)
    return sum over cells and channels of  weight * (field - target)^2
```

The target is `field` evaluated once at the true parameters and then frozen. Cells
where the observable is undefined are excluded, and the valid set comes from the
target, so the loss domain does not move while the parameters do.

```
calibrate(start, iterations):
    target, weights <- loss ingredients at the true parameters
    g0    <- gradient of the loss at the start point
    clip  <- clip_factor * norm(g0)                     # units differ per observable
    opt   <- adam(cosine decay from learning_rate to 0 over `iterations`)
    for each iteration:
        loss, gradient <- value_and_grad(loss)(params)  # one pass through the model
        gradient       <- clip_by_global_norm(gradient, clip)
        params         <- adam update
        append (params, loss, gradient) to the trajectory
```

One `jax.value_and_grad` call differentiates the whole coupled system; no adjoint is
written by hand. Two details matter for reading the output:

- **Segmented rollout.** `Coupler.run` returns only its final state, so averaging the
  last three days needs three chained couplers, `[8, 1, 1]` for a 10-day rollout.
  Chaining is function composition, so the gradient still crosses the whole rollout.
- **Trajectory rows.** Row `k+1` holds the parameters produced *after* iteration `k`
  together with the loss evaluated *before* it. The lowest-loss iterate's parameters
  are therefore read from row `k`, not `k+1`.

The gradient clip is measured at each run's own start point rather than fixed, because
an MLD loss is in m^2 and a T/S loss is variance-normalised; their gradient norms
differ by orders of magnitude and one fixed threshold would either never fire or fire
every step. Clipping is there because an unclipped run once produced a gradient ~200x
its neighbours, which adam's momentum carried for some 25 iterations. The default
factor of 1.2 is a round value above the start-point norm; it rarely binds, firing on
a handful of the 200 iterations. Reports 22 and 23 used 1.2205, which was an earlier
hand-picked threshold expressed as a multiple of that report's start-point norm --
pass `--clip-factor 1.2205` to reproduce them exactly.

### Observables

`mld_avg` is the depth at which potential density first exceeds its value at the
reference level by 0.03 kg/m^3, linearly interpolated between the two bracketing
levels. The discrete level selection is kept off the gradient path and the
interpolation's division is guarded, so degenerate columns cannot poison the gradient.

`tsdiff_avg` is four channels, `T[-2]-T[-1]`, `T[-3]-T[-1]`, `S[-2]-S[-1]` and
`S[-3]-S[-1]`. On this 15-level grid the mixed-layer depth is a function of exactly
these levels, so this is the same information without the density combination. Each
channel is weighted by `1 / (cells * variance)` of its own target channel, which makes
the four terms comparable; the weights come from the target alone, so every evaluation
shares one loss.

## Reproducing the results

The landscape scan splits by `c_k` column so that four jobs can share it:

```bash
for i in 0 4 8 12; do
    python landscape.py scan --observable tsdiff_avg \
        --col-start $i --col-end $((i + 4)) --out grid_tsdiff/ &
done; wait
python landscape.py merge --out grid_tsdiff/
```

Four descents from the corners of a box around the truth (`c_k` = 0.1 +/- 50%,
`c_eps` = 0.7 +/- 0.3), fixed before the runs:

```bash
python calibrate.py --observable tsdiff_avg --c-k 0.05 --c-eps 0.40 --out runs_tsdiff/start_0
python calibrate.py --observable tsdiff_avg --c-k 0.15 --c-eps 1.00 --out runs_tsdiff/start_1
python calibrate.py --observable tsdiff_avg --c-k 0.05 --c-eps 1.00 --out runs_tsdiff/start_2
python calibrate.py --observable tsdiff_avg --c-k 0.15 --c-eps 0.40 --out runs_tsdiff/start_3
```

Fields for the snapshot figure, from the run that recovered the parameters:

```bash
python snapshot.py --run runs_tsdiff/start_3 --c-k 0.15 --c-eps 0.40
```

Figures:

```bash
python figures/optimization.py --grid grid_tsdiff/ \
    --run runs_tsdiff/start_0 --run runs_tsdiff/start_1 \
    --run runs_tsdiff/start_2 --run runs_tsdiff/start_3 \
    --out figures/optimization.png
python figures/snapshot.py --run runs_tsdiff/start_3 --out figures/snapshot.png
```

Pass `--observable mld_avg --loss-units 'm$^2$'` for report 22's observable; the
commands are otherwise identical.
