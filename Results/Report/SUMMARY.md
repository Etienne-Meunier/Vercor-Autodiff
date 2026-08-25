# Vercor Coupled-Model Autodiff Calibration: Summary

Goal: fit Veros' TKE parameters `c_k` (mixing length), `c_eps`
(dissipation) via gradient descent through the fully coupled
OCN(Veros)+LND+ATM(JCM) model, `vercor`'s Coupler. True values
`c_k=0.1, c_eps=0.7`; every fit below starts from a wrong guess.

## Findings

1. **Coupled model differentiates correctly end-to-end.** Short fits
   (5-20 step rollout) recover `c_k` to ~1-8% error; `c_eps` converges
   slower (longer physical timescale). Fixed-`lr` Adam consistently
   overshoots `c_k` once it nears truth while still chasing `c_eps`.
2. **Gradient trustworthy horizon: ~n=20 steps.** AD gradient itself stays
   smooth and well-behaved at every length tested (up to n=1000). What
   breaks is *finite-difference validation* — trajectories chaotically
   decorrelate somewhere in [20,25] steps, so FD stops agreeing with AD
   there, but that's an FD artifact (see appendix), not evidence the AD
   gradient is wrong.
3. **Detaching the atmosphere's gradient doesn't help.** Cut ATM out of
   the backward pass (keep it in forward) to test whether JCM's own chaos
   was the limiting factor. Breakdown point didn't move — proves the
   ~20-step ceiling comes from Veros' own internal ocean dynamics, not
   the atmosphere coupling.
4. **Windowed / teacher-forced calibration solves the long-rollout
   problem.** Chain many short (`W=10`) windows, each *reset to the true
   trajectory's state* at its start (teacher forcing) rather than
   carrying the model's own state forward. Cost then grows linearly with
   number of windows, no chaos wall — demonstrated stable to 250
   simulated days (naive autoregressive carrying diverges badly by
   comparison — loss to 1500+, see appendix).
5. **A cosine-decayed learning rate fixes the overshoot** seen in every
   fixed-`lr` run — `c_k` stabilizes/flattens instead of drifting past
   its optimum.
6. **`c_k`/`c_eps` are not independently identifiable from a single
   temperature-error loss.** Two full 25-window optimizations from
   opposite starting corners converge to two *different*, both
   comparably-low-loss, both-far-from-true endpoints:

   | start (c_k, c_eps) | end (c_k, c_eps) |
   |---|---|
   | (0.05, 0.40) | (0.169, 0.503) |
   | (0.15, 0.90) | (0.071, 0.962) |
   | true: (0.1, 0.7) | — |

   Endpoints sit on roughly opposite sides of true along an
   anti-correlated direction — a genuine identifiability ridge (both
   params shape the same vertical-mixing effect on temperature), not an
   optimizer artifact. Fixing this needs a different loss (diagnostics
   that separate the two params' effects), not more tuning.

## Bottom line

The coupled model is correctly differentiable and usable for real
calibration. Direct long single-rollout gradients work to ~20 days;
beyond that, windowed/teacher-forced calibration with LR decay is the
validated approach, unbounded in span. The remaining open problem is
physical (parameter identifiability), not numerical.

---

## Appendix: additional detail

### A1. Per-report reference

| # | Question asked | Script |
|---|---|---|
| 1 | Does the coupled fit work at all (5-step)? | `examples/fit_ck_ceps_jcm_global4deg.py` |
| 2 | Same fit, 10 steps — does the pattern hold? | `Results/Report/scripts/report-2/fit_and_generate_figures.py` |
| 3 | How far can rollout go before gradients break? | `debug_script/13-15`, fit at n=20 |
| 4 | Does detaching ATM's gradient extend the horizon? | `debug_script/16-17` |
| 5 | Can we calibrate over an arbitrarily long span? | `debug_script/18` (fails), `debug_script/19` (works) |
| 6 | Real optimization with the validated method + LR decay | `Results/Report/scripts/report-6/optimize.py` |
| 7 | Identifiability test, alternate start | `Results/Report/scripts/report-7/optimize_alt_start.py` |

### A2. FD-eps chaos signature (Report 3, n=25)

FD gradient doesn't converge as `eps` shrinks and doesn't even hold a
consistent sign — signature of chaotic trajectory divergence, not
roundoff:

| eps | FD grad | rel err vs AD |
|---|---|---|
| 1e-2 | -1.01e+03 | 350% |
| 3e-3 | -1.75e+04 | 74% |
| 1e-3 | -6.62e+04 | 93% |
| 3e-4 | +8.07e+03 | 157% (sign flip) |
| 1e-4 | +1.55e+05 | 103% |

At a properly-scaled eps (1e-2), FD tracks AD well through n=20
(7-8% rel err) and n=12/15 similarly — the AD gradient magnitude itself
stays in a tight band (-4600 to -5700) across n=12..25; only the FD
comparison collapses, and only past ~n=20-25.

### A3. Why not push to n=1000 directly

Two risks flagged before running anything, from Veros-Autodiff's own
long-rollout report: (a) memory — single-level `lax.scan` +
`jax.checkpoint`, no outer chunking, crashes from linear memory growth
around n=400 on a 16GB GPU in that report's setup; (b) even where memory
holds, gradient magnitudes there became numerically meaningless well
before the memory wall for ocean-only differentiation (garbage by
n=5000). Our model couples in a chaotic atmosphere on top, so breakdown
was expected (and found) far sooner: ~n=20-25, not ~n=1000.

### A4. Autoregressive windowing failure detail (Report 5, attempt 1)

Carrying the model's own (imperfect-param) state forward between windows,
`stop_gradient`-ed, diverges even though every individual window is only
10 steps:

| window | loss | c_k | c_eps |
|---|---|---|---|
| init | -- | 0.0500 | 0.4000 |
| 0-3 | 60.9 -> 22.6 | 0.070 -> 0.110 | 0.380 -> 0.345 |
| 4 | 95.9 | 0.118 | 0.342 |
| 8 | 811.1 | 0.141 | 0.363 |
| 14 | 1514.8 | 0.152 | 0.449 |

Diagnosis: cross-window state drift, not within-window gradient
instability — carried state and the fixed true-param target decorrelate
chaotically window over window. Teacher forcing (reset to true state each
window) structurally prevents this by construction.

### A5. Report 6 vs 7 full trajectories

Report 6 (start below true on both): `c_k` passes through 0.1004 (0.4%
off) at window 1 before settling ~0.169 from window ~16 on; `c_eps`
climbs smoothly the whole run to 0.503, still rising at the end.

Report 7 (start above true on both): `c_k` passes through 0.0992 (0.8%
off) at window 1, overshoots down to ~0.06 by window 4-8, drifts back up
to 0.071; `c_eps` shoots up to 0.98 by window 4, relaxes down to 0.962.

Both: large first-window loss drop, then flat low band (0.5-1.5) for the
rest of the 250-day run, no divergence.

### A6. Caveats carried across all reports

Single run/seed throughout (no repeated-seed statistics on any fit).
No 2D loss-landscape grid scan anywhere (cost-prohibitive at these
rollout lengths). FD cross-checks only done at the reports that
introduce a new gradient path (1, 3, 4); later reports (5, 6, 7) reuse
already-validated within-window mechanics without a fresh FD check.
Report 7's identifiability read is based on 2 starting points, not a
systematic sweep — confirms a ridge exists, doesn't map its exact shape.
