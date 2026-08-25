# Stable Calibration Over Arbitrarily Long Simulated Spans: Windowed/Teacher-Forced Gradients

Direct answer to "can we calibrate over a longer rollout in a stable way,
even approximate, as long as we can keep optimizing": **yes** -- chain many
short (within-trustworthy-horizon) differentiated windows instead of one
long one. Two variants tried; only one works, and the reason it fails vs.
succeeds is itself informative.

Scripts: `debug_script/18_windowed_tbptt_calibration.py` (autoregressive,
fails), `debug_script/19_windowed_teacher_forced.py` (teacher-forced,
works).

## The idea

Same idea as truncated backpropagation through time (TBPTT) for training
RNNs on long sequences, or "cycling"/windowed adjoint methods in
operational data assimilation for chaotic geophysical systems: never
differentiate more than `W` steps at once (Report 3/4 found gradients
trustworthy through `n<=20`; `W=10` used here for margin), but *chain* many
such windows so the total simulated span and the total number of
optimizer updates can be as large as desired -- cost grows linearly with
the number of windows, not superlinearly, and no single gradient call ever
re-enters the chaotic-breakdown regime.

Both variants below use `W=10`, `N_WINDOWS=15` (150 total simulated days),
`optax.adam(lr=2e-2)`, same true params (`c_k=0.1, c_eps=0.7`) and wrong
start (`c_k=0.05, c_eps=0.4`) as Reports 1-3, one gradient update per
window (15 updates total).

## Attempt 1: autoregressive carry (fails)

Each window's *forward* result (computed with the current, still-wrong
params) becomes the next window's starting state, `stop_gradient`-ed so
the next window's backward pass doesn't reach through it. This is the
literal TBPTT recipe for RNNs.

| window | loss | c_k | c_eps |
|---|---|---|---|
| init | -- | 0.0500 | 0.4000 |
| 0-3 | 60.9 -> 22.6 | 0.070 -> 0.110 | 0.380 -> 0.345 |
| 4 | 95.9 | 0.118 | 0.342 |
| 8 | 811.1 | 0.141 | 0.363 |
| 14 | **1514.8** | 0.152 | 0.449 |

Windows 0-3 look exactly like Report 1-3's clean short-window behavior
(loss dropping, `c_k` approaching true). Then it diverges -- **not** from
within-window gradient instability (each window is still only 10 steps),
but from **cross-window trajectory drift**: the carried-forward state
evolves under whatever params were current at that point, which are
imperfect until convergence. Since this is a chaotic system, even a small
residual param error means window `i`'s carried state and the
pre-computed (fixed, true-param) target for window `i` are two nearby
points on a chaotic attractor that have already decorrelated by the time
window `i` runs -- the same exponential trajectory divergence from Reports
3-4, just relocated to operate *between* windows instead of *within* one.
Comparing decorrelated states produces a loss/gradient that reflects
attractor geometry noise, not parameter sensitivity, and Adam's momentum
compounds the resulting bad updates.

## Attempt 2: teacher forcing (works)

Instead of carrying the model's own state forward, reset every window's
starting state to the **true** trajectory's state at that window's start
(precomputed once, alongside the targets). Standard technique in
sequence-model training ("teacher forcing") and cycling data assimilation.
Every window becomes an independent, clean W-step calibration problem that
always starts from ground truth -- cross-window error compounding is
structurally impossible.

| window | loss | c_k | c_eps |
|---|---|---|---|
| init | -- | 0.0500 | 0.4000 |
| 0 | 6.088e+01 | 0.0700 | 0.3800 |
| 1 | 4.882e-01 | 0.0837 | 0.3664 |
| 2 | 1.767e+00 | 0.0936 | 0.3659 |
| 3 | 6.187e-01 | 0.1016 | 0.3674 |
| 5 | 1.153e+00 | 0.1134 | 0.3799 |
| 8 | 6.186e-01 | 0.1252 | 0.4047 |
| 11 | 6.432e-01 | 0.1331 | 0.4281 |
| 14 | 9.190e-01 | 0.1386 | 0.4485 |

**Loss stays bounded and low across all 15 windows** (0.49-1.77 after the
first window, vs. attempt 1's runaway to 1514.8) -- no divergence, no
NaN, no sign of chaos re-entering. `c_k` passes almost exactly through
the true value at window 3 (0.1016, 1.6% off) before continuing to climb
(reaching 0.1386 by window 14) -- the same overshoot-past-optimum pattern
Reports 1-3 saw with `optax.adam`'s fixed learning rate and no decay, now
compounding over 15 consecutive same-direction updates instead of 5-10.
`c_eps` climbs steadily and monotonically toward true (0.38 -> 0.45),
exactly the slow-convergence behavior already characterized in Report 1.

## Why this answers the original question

- **The total simulated span is no longer bounded by chaos.** 150 days
  here, and there's nothing chaos-related stopping it from being 1500 or
  15000 -- cost is linear in the number of windows (each one ~90s for a
  gradient step + ~16s for its precomputed target here), not something
  that hits a wall like the single-long-rollout approach did at n~20-25.
- **The gradient computed in any one window is exactly as trustworthy as
  Report 3 found short windows to be** -- nothing new is happening inside
  a window; the whole point was avoiding ever re-entering the untrustworthy
  regime.
- **It's an approximation, not the true long-horizon gradient** -- as
  flagged when this was proposed, teacher forcing means each window
  evaluates local sensitivity to the params *given the correct state*, not
  the compounded effect of imperfect params on where the state ends up
  d(loss)/d(params) `truly` over the full window sequence. That's exactly
  the "even approximate is fine" tradeoff accepted going in.

## Remaining rough edge: fixed learning rate overshoot

Both this result and every earlier report show the same signature: a
fixed `lr=2e-2` Adam step overshoots `c_k`'s optimum once it's reached
(clearest here, since 15 consecutive updates keep pushing it the same
direction well past the 1.6%-error point at window 3). Not attempted here,
straightforward next step: a learning-rate schedule (decay after loss
stops improving, or per-parameter rates so `c_k` slows down while `c_eps`
-- which is far from converged the whole time -- keeps its full step size).

## Caveats

Single run, no repeat seeds, one gradient update per window (not multiple
inner iterations per window before advancing) -- both variants could be
explored further within the same windowed framework. No finite-difference
cross-check of the teacher-forced gradient itself in this report (each
window's gradient mechanics are identical to Report 3's already-validated
`n<=20` case, so a fresh check wasn't run, but would be cheap to add for
extra confidence).
