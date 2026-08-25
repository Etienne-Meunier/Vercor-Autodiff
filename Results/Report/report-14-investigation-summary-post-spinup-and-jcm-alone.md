# Report 14: Investigation Summary -- Post-Spinup Gradients, eps-Scale, and JCM-Alone Differentiability

Consolidates the overnight follow-up investigation into report-10/11's surprising finding (spin-up makes gradients *worse*, not better) and the question it raised: a referenced external paper reportedly does stable JCM calibration over 5 steps, which seemed inconsistent with report-11's n<=2 post-spinup horizon. Three threads, one resolved cleanly, one still open.

## Thread 1 (resolved): report-11's n<=2 threshold was itself an eps-scale artifact

Report 13's eps-sweep at post-spinup n=3 hinted the correct FD eps might be much smaller than the eps=1e-2 used throughout reports 1-11 (established as the right scale for *cold-start* windows). Reran report-11's full n-bracket at eps=1e-4 instead of eps=1e-2, same 50-day-spun-up state:

| n | AD grad | FD grad (eps=1e-2) | rel err (eps=1e-2) | FD grad (eps=1e-4) | rel err (eps=1e-4) |
|---|---|---|---|---|---|
| 1 | 1.0871e+02 | 1.0958e+02 | 0.8% | 1.0871e+02 | 0.0% |
| 2 | 2.7093e+02 | 2.7445e+02 | 1.3% | 2.7099e+02 | 0.0% |
| 3 | 3.9469e+02 | -4.2036e+02 | 193.9% | 3.9487e+02 | 0.0% |
| 4 | 4.0970e+02 | -7.5099e+02 | 154.6% | 4.1004e+02 | 0.1% |
| 5 | 2.7772e+02 | -9.5789e+02 | 129.0% | -1.6211e+05 | 100.2% |
| 6 | 8.4037e+01 | 9.3542e+02 | 91.0% | -2.9996e+05 | 100.0% |
| 8 | -2.6539e+02 | 4.5783e+02 | 158.0% | -2.0256e+05 | 99.9% |
| 10 | -3.2886e+01 | -2.9510e+03 | 98.9% | 2.8154e+04 | 100.1% |
| 20 | -1.4937e+03 | -1.1930e+04 | 87.5% | -2.8142e+06 | 99.9% |

At the correct eps, the trustworthy window is **n<=4** (rel err <=0.1%), breaking sharply and cleanly between n=4 and n=5 -- not n<=2 as report-11 first found. eps=1e-2 was simply too large for the post-spinup regime's much-tighter linear-perturbation radius, producing spuriously large errors even at n=3-4 where the gradient is actually fine. Past n=4, FD blows up to huge magnitudes (1e5-1e6) while AD stays modest -- rel err pins near 100% because AD becomes negligible relative to a diverging FD, not because AD itself is unstable.

**Revised bottom line**: post-spinup trustworthy horizon is n<=4, still ~5x shorter than cold-start's n<=20 (report-3), but not the ~10x report-11 first suggested. The core finding from report-10/11 stands: starting from an already-turbulent atmosphere genuinely shrinks the usable window, it's just n<=4 rather than n<=2.

## Thread 2 (blocked, real bug found and partially fixed): JCM-alone (no full Veros ocean) gradients

Built a JCM+toy-slab-ocean+toy-slab-land configuration (`examples/run_jcm_with_slab.py`'s setup, matching the "JCM alone" comparison point) to test whether removing full Veros ocean dynamics explains why the referenced paper's n=5 works. Two bugs found in sequence:

**Bug 1 (fixed)**: `report-12/jcm_alone_bracket.py`'s first version perturbed `jcm_state.prog.temperature` and got exactly zero AD and FD gradient at every n. Traced the cause to `vercor/setups/_external/jax_gcm_state.py:131-136`: the step function integrates `state.metadata` (dinosaur's spectral `primitive_equations.State`), not `state.prog` -- `prog` is a nodal-space diagnostic *recomputed from* `metadata` after each step, not an input to the dynamics. Perturbing it is a structural no-op. Fixed by perturbing `metadata.temperature_variation` (the real spectral prognostic field) instead; verified with a direct forward check (perturbed mean 293.78 vs baseline 288.0 after one step -- confirms the perturbation now actually propagates).

**Bug 2 (found, not fixed)**: with the corrected perturbation target, FD is finite and smooth (2.45e7 at n=1 down to 4.38e6 at n=20, monotonic), but **AD returns NaN at every single n from 1 to 20**. Diagnosed with `jax_debug_nans`: the NaN originates in the backward pass (transpose) of a `lax.scan` primitive inside JCM's `model.run_from_state` (JAX's own transpose-rule dispatch, `_scan_transpose_fancy` -> `FloatingPointError: invalid value (nan) encountered in scan`). `jax_debug_nans` doesn't attribute further inside the scan body at this dispatch level (the whole scan-transpose is one primitive call from its perspective) -- pinpointing the exact physics operation inside JCM's ~30-minute-substep integration would need unrolling the scan into a Python loop (loses the compile-once benefit, or needs a scoped `jax.disable_jit()`/manual step-by-step bisection) or instrumenting JCM's physics scheme directly, both out of scope for tonight.

**Status**: this specific standalone JCM+slab wiring in vercor has a genuine differentiability gap unrelated to physical chaos -- the question "does the referenced paper's n=5 JCM-alone result reproduce here" is **unresolved**, not negative. The forward simulation and FD both work fine; only the AD backward pass through this particular state-injection path fails.

## What this means together

- The apparent contradiction ("their paper does n=5, we can't even do n=2") is now explained differently than either extreme: our own coupled-model post-spinup horizon is n<=4 (thread 1), which is *close to* the referenced n=5, not off by 10x as first thought.
- Whether pure JCM+slab (without full ocean chaos) would do noticeably better than n<=4 is still unknown -- thread 2 hit a plumbing bug before any physics comparison was possible.
- The recurring lesson across reports 3, 11, and this one: **FD eps must be re-checked whenever the base state changes** (cold start vs. post-spinup use different appropriate eps scales) -- a single fixed eps convention is not safe across regimes, worth remembering for any future bracket.

## Suggested next steps (not done tonight)

1. Fix bug 2: get precise attribution for the NaN in JCM's scan-transpose (start by disabling `jitted` throughout, matching debug_script/21's unjitted setup, then binary-search which physics substep introduces it -- likely a sqrt/log/division near a physically-degenerate value triggered by the perturbed spectral state).
2. Once fixed, rerun report-12's bracket (n=1..20) to get the real JCM-alone AD-vs-FD comparison and see whether it extends meaningfully past coupled-model's n<=4.
3. Consider whether report-3's original cold-start eps=1e-2 bracket is itself slightly conservative -- not checked here, but thread 1's finding (eps sensitivity varies by regime) makes it worth a similar eps=1e-3/1e-4 spot-check at cold-start n=20-25 for completeness.

## Caveats

Single spin-up length (50 days) and single seed throughout. Thread 1's n=4/n=5 breakpoint not bracketed finer (no n at 4.5-equivalent). Thread 2's bug-2 root cause is diagnosed to the scan-transpose level, not the exact physics line -- "likely a masking/degenerate-value issue" is an informed guess, not confirmed.
