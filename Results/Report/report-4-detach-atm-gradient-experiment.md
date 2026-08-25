# Does Detaching the Atmosphere's Gradient Extend the Usable Rollout?

Follow-up to Report 3's finding that gradients through the fully-coupled
OCN+LND+ATM(jcm) model become chaotically unreliable somewhere between
n=20 and n=25. Hypothesis: since JCM is a chaotic weather model with a
much shorter predictability horizon than ocean dynamics, maybe cutting the
gradient path *through* the atmosphere (while keeping it in the forward
simulation) would let gradients stay valid much longer.

**Result: no, it doesn't help.** The chaos onset is at essentially the
same rollout length with or without the atmosphere in the backward pass —
strong evidence the ocean's own internal dynamics, not the atmosphere
coupling, sets the practical ceiling for this model/setup.

## Method

`debug_script/16-17`: monkeypatched
`vercor.setups._external.veros_fluxes.compute_fluxes` to wrap its
`runtime_fields` argument (the ATM->OCN exchanged forcing: velocity,
humidity, model level height, density, potential temperature, temperature,
radiation fluxes) in `jax.lax.stop_gradient` before Veros' bulk-flux
formulas consume them. Forward values are unchanged (`stop_gradient` is
the identity forward); only the backward graph changes -- `d(loss)/d(c_k)`
no longer threads through JCM's dynamics at all, only through OCN's own
repeated forcing-application physics across steps.

Single `value_and_grad(c_k)` + FD (`eps=1e-2`, the scale that worked well
in Report 3's bracket) at N_STEPS in [20, 30, 40, 50].

## Results

| N_STEPS | AD grad (detached) | FD grad (eps=1e-2) | rel err |
|---|---|---|---|
| 20 | -3.64e+03 | -5.10e+03 | 28.6% |
| 30 | -3.22e+03 | -1.36e+03 | 137.8% |
| 40 | -2.76e+03 | -4.90e+02 | 463.4% |
| 50 | -2.45e+03 | **+6.05e+04** | 104.0% (sign flip) |

Compare to Report 3's full-coupled (non-detached) bracket at the same
lengths: n=20 rel err was 7.3%, and the full-coupled sweep had already
broken down by n=25 (93% rel err, sign-unstable across eps). **The
detached version's onset (clearly broken by n=30) is if anything earlier**,
not later, than the full-coupled version's (broken by n=25) -- within the
noise of a single-seed comparison, but certainly not the order-of-magnitude
extension the hypothesis predicted.

## Interpretation

Two things stand out:

- **The AD gradient itself never blows up** -- it decreases smoothly and
  monotonically in magnitude across the whole sweep (-3643 -> -3224 ->
  -2760 -> -2445). Whatever is happening, it isn't runaway gradient
  explosion in the differentiated quantity itself.
- **FD validity is what collapses**, and it collapses at essentially the
  same rollout length regardless of whether the atmosphere is in the
  backward graph. Since detaching ATM removes its entire contribution to
  the gradient (real feedback terms and all), and the breakdown point
  didn't move, the chaotic sensitivity responsible for FD's collapse must
  live in **OCN's own internal dynamics** -- baroclinic instability,
  TKE-mixing-affecting-buoyancy-affecting-circulation feedback, or similar
  -- not in the ocean-atmosphere coupling. Veros' intrinsic chaotic
  timescale in this 4-degree global configuration appears to be on the
  order of 3-4 weeks, independent of whether a weather-chaotic atmosphere
  is coupled in at all.

This is a useful negative result: it rules out the most obvious mitigation
(cut the atmosphere out of the backward pass) and redirects where a real
fix would need to focus -- on the ocean's own dynamics, not the coupling.

## What this means for the ~20-step ceiling

The practical rollout-length ceiling found in Report 3 (trustworthy
through n=20, unreliable by n=25) is not an artifact of atmosphere
coupling that a cheap gradient-truncation trick can route around. Options
for genuinely extending it, none attempted here (all real efforts of their
own):

- **Ensemble/covariance-based calibration** (e.g. ensemble Kalman
  inversion) -- average gradient-like signal over many perturbed
  realizations to cancel chaotic noise, at the cost of needing many
  forward (and possibly backward) rollouts per calibration step instead
  of one.
- **Time-averaged loss diagnostics** instead of instantaneous full-field
  temperature (the Veros-Autodiff long-rollout report's `mld_ma` was an
  attempt at this, though it had its own unrelated bug at the time of that
  report) -- averaging over a window may smooth out chaotic noise that an
  instantaneous snapshot loss doesn't.
- **Accept the ~20-day ceiling** as the practical regime for this exact
  single-realization gradient-based method, and calibrate using several
  independent short (~10-20 step) windows rather than one long one.

## Caveats

Single seed/realization throughout (here and in Report 3) -- the exact
breakdown point (20 vs. 25 vs. 30) could shift somewhat with different
initial conditions or `c_k` test values; the qualitative finding (detaching
doesn't meaningfully move the ceiling) is the robust part of this result,
not the precise step count. No full fit run with the detached gradient at
n=20 (where it still roughly tracks FD, 28.6% error) -- given it offers no
real advantage over the full-coupled gradient at that length (which was
more accurate there, 7.3%), there's no reason to prefer the detached,
biased version for that regime.
