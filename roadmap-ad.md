# Roadmap: differentiable Veros (autodiff fork) integration

Goal: a differentiable Veros ocean component, coupled through vercor's own
`Coupler`, that exposes `jax.grad`/`jax.jit` over a multi-step rollout the
same way `examples/differentiate_jax_component.py`-style scripts do for a
pure-JAX component. Starting point: the ACC setup (proven differentiable in
`Veros-Autodiff/notebooks/demonstration/gradient-computation.ipynb`).

## Status (2026-07-31): core wiring done, verified end to end

The plan below was written before implementation and described an
**additive**, parallel-files approach (new `*_jax` adapter trio, a separate
`VerosJaxConfig`, a new `configure_veros_runtime_jax()`). That's not what
happened — the existing adapter was extended **in place** instead, behind a
single config switch, and it works:

- `configure_veros_runtime()` (`veros_runtime_settings.py`) now sets
  `backend="jax"`/`linear_solver="scipy_jax"` unconditionally — there's no
  separate numpy-vs-jax runtime-config function, just one, changed.
- `VerosConfig` (`vercor/setups/config.py`) gained `setup:
  Literal["global_4deg", "acc", "global_4deg_learning"]` and
  `uses_atmosphere_forcing: bool`, and replaced the old `jitted: bool` field
  with `execution: Literal["jax", "host"] = "jax"`. `make_veros_gcm` derives
  `jitted = (execution != "host")` and passes `execution` straight through to
  `ComponentSpec`, so the two concerns (jax-scanned dispatch, and whether the
  Veros step itself is jit-wrapped/pytree-copy-safe) can no longer disagree.
  This matters: an earlier `jitted=False` + `execution="jax"` combination
  (independent flags, briefly possible mid-refactor) reproducibly hung —
  unifying them into one flag removed that state entirely.
- `ACCSetup` (`vercor/setups/_external/veros_setup_acc.py`) and
  `GlobalFourDegreeLearningSetup`
  (`vercor/setups/_external/veros_setup_global4deg_learning.py`) are both
  ported in and selectable via `VerosConfig(setup=...)`, dispatched through
  `VEROS_SETUP_FACTORIES` in `veros_gcm_state.py` — matches section 3 below
  as originally planned.
- `copy_state` (`veros_state.py`) was rewritten to use `VerosState.copy()`
  instead of a deepcopy-based rebuild through the constructor, specifically
  because rebuilding allocated fresh `VerosSettings`/`var_meta` objects that
  broke pytree-structure stability across `lax.scan` iterations. This is the
  "open risk" flagged in the original section 4 below — it needed solving,
  not skipping, because the host adapter's `pure`/`copy_state`/solver-cache
  machinery is exactly what still bridges Veros into the coupler under
  `execution="jax"` too (no parallel jax-only version was built; the
  existing machinery was made scan-safe instead).
- `backends.py`'s scanned executor now wraps the per-step body in
  `jax.checkpoint` before `jax.lax.scan` — remat is live.
- Example scripts landed as `examples/run_verosad.py` (forward-only sanity
  check against the ACC setup), `examples/run_verosad_global4deg.py`
  (ERA5-coupled, mixed `execution="host"` ATM/LND + `execution="jax"` OCN —
  section 7's plan, confirmed working), and `examples/run_verosad_grad.py`
  (differentiable ACC rollout, `jax.value_and_grad` vs. finite-difference
  check) — instead of the single planned `run_veros_acc_diff.py`.
- Verified in `run_verosad_grad.py`: `jax.value_and_grad` through
  `cpl.run(...)` on the ACC setup, `execution="jax"`, matches a
  finite-difference gradient to within ~0.3% at 2 steps and ~1.7% at 50 steps
  (the latter gap is consistent with `eps=1e-2` truncation error over a
  longer, more nonlinear rollout, not an AD bug).

## Open follow-ups

- **No cross-call compilation caching in `cpl.run()`.** `execute_plan`
  (`vercor/_runtime/execution.py:167`) builds its `jax_executors` cache dict
  fresh inside every call, so calling `cpl.run(...)` bare and repeatedly
  (e.g. for a finite-difference check, or a multi-iteration training loop)
  recompiles the whole scanned executor from scratch every time — confirmed
  by direct timing (~5s every call, no speedup on repeats). The workaround
  the example scripts use is to wrap the *entire* call chain — including
  `cpl.run` — in one persistent `jax.jit` created once and reused (as
  `grad_fn = jax.jit(jax.value_and_grad(loss_fn))` does); calling `cpl.run`
  bare anywhere in a hot loop silently pays this cost every time. Worth
  either documenting this contract loudly on `Coupler.run`/`RuntimeOptions`,
  or giving the Coupler its own persistent executor cache so bare repeated
  calls aren't a trap.
- **Per-step `jax.jit` may now be redundant.** `VerosGCMSetupState`
  jit-wraps `_step_function` whenever `jitted=True` (i.e. whenever
  `execution="jax"`), but that step now also runs inside
  `backends.py`'s outer `jax.jit(lax.scan(jax.checkpoint(...)))`. Nesting a
  jit inside a checkpointed scan body isn't known to be wrong, but it's
  unproven whether it helps, hurts compile time, or changes remat
  granularity — worth a profiling pass, though not blocking.
- **`apply_veros_forcing_fields`** (`veros_state.py`) now branches on
  `isinstance(updated, jax.core.Tracer)` to skip `runtime_array_to_host`
  while tracing — needed to keep the ERA5 atmosphere-forcing path
  (`uses_atmosphere_forcing=True`, e.g. `global_4deg_learning`) trace-safe
  under `execution="jax"`. Worth double-checking this doesn't mask a case
  where host-array conversion should still happen (e.g. `execution="host"`
  called from inside someone else's outer `jax.jit`).

## 1. Point the interpreter at the fork, not pip's `veros` — done

- Examples run with the fork on `PYTHONPATH`, e.g.:
  `sys.path.insert(0, "/Users/emeunier/Desktop/Projets/Veros-Autodiff/veros")`
  at the top of each `examples/run_verosad*.py` script — same trick
  `Veros-Autodiff/scripts/load_runtime.py` uses.
- `make_veros_gcm`'s `try: import veros` guard is unaffected — it doesn't care
  which `veros` it gets, only that one is importable.

## 2. Runtime config — done, but as a modification, not an addition

- `configure_veros_runtime()` itself now sets `backend="jax"`,
  `linear_solver="scipy_jax"` (`veros_runtime_settings.py`). There is no
  separate `configure_veros_runtime_jax()` — the original numpy-backend path
  no longer exists as an option. If a numpy/host-numpy fallback is ever
  needed again, it would have to be reintroduced deliberately.

## 3. Config-selectable setup class (topography choice) — done

- `VerosConfig.setup: Literal["global_4deg", "acc", "global_4deg_learning"]`
  in `vercor/setups/config.py`, dispatched via `VEROS_SETUP_FACTORIES` in
  `veros_gcm_state.py`. `ACCSetup` ported from the disabled-diagnostics
  `acc_learning.py` variant as planned.

## 4. Adapter machinery — extended in place, not paralleled

- No new `veros_gcm_state_jax.py` / `veros_runtime_jax.py` / `veros_gcm_jax.py`
  files. The existing trio now branches on `config.execution` instead.
- `copy_state`/`pure`/`get_component_linear_solver` in `veros_state.py` were
  **not** skipped — they were fixed to be scan-safe instead (see Status
  above) and are still exactly what `pure()` uses to bridge into Veros'
  native step under `execution="jax"`.
- `execution="jax"` on the `ComponentSpec`, `VerosState` passed straight
  through as the component `payload` — confirmed working.
- **Former open risk, now resolved by testing:** the linear-solver caching
  (`get_component_linear_solver`) is still in use and does work correctly
  traced once inside `lax.scan` — `run_verosad_grad.py` verifies this
  end-to-end (gradient matches finite difference).

## 5. Output — unchanged, still the plan

- `coupler.run(state, output=None)` inside the jit/grad path, as
  `run_verosad_grad.py` does.
- For periodic snapshots/inspection, a separate non-jitted, non-diffed
  `coupler.run(..., output=OutputTarget(...))` call outside the gradient
  computation remains the intended pattern — not yet exercised in an
  example script.

## 6. Grid/regridding — unchanged, still accurate

- `VerosGCMSetupState.grid` is built from the live Veros state's own
  `xt`/`yt`/`maskT`, not hardcoded to the 4-degree topology. Swapping setups
  changes `nx, ny, nz`/topography and the `bilinear` regridder factory
  re-derives its weights automatically.

## 7. Mixed host/jax coupling — done, `run_verosad_global4deg.py`

- ATM/LND (`make_era5_atmosphere`/`make_era5_land`) stay `execution="host"`
  while OCN is `execution="jax"` via `VerosConfig(setup="global_4deg_learning",
  uses_atmosphere_forcing=True, ...)` (default `execution="jax"`). Matches
  the original plan: gradient flow only needs to trace forward through
  ATM into OCN as data, never backward through ATM.

## 8. Example scripts — done, different names than planned

- `examples/run_verosad.py`: forward-only ACC rollout sanity check
  (`uses_atmosphere_forcing=False`), no differentiation.
- `examples/run_verosad_global4deg.py`: full ERA5-coupled mixed host/jax run
  (section 7).
- `examples/run_verosad_grad.py`: differentiable ACC-only rollout —
  `jax.value_and_grad(loss_fn)` wrapped in an outer `jax.jit`, validated
  against a finite-difference gradient. This is the one that actually proves
  the differentiability goal from the top of this doc.

## What's reuse vs. genuinely new work (updated)

- **Reuse, untouched:** ERA5 atmosphere/land components, `Exchange`/`bilinear`
  regridding, `SurfaceMaskPolicy`, `Coupler`/`Clock`.
- **Modified in place (not additive):** `veros_runtime_settings.py`,
  `veros_gcm.py`, `veros_gcm_state.py`, `veros_runtime.py`, `veros_state.py`,
  `vercor/setups/config.py`, `vercor/_runtime/backends.py`.
- **New, additive files:** `veros_setup_acc.py`,
  `veros_setup_global4deg_learning.py`, three `examples/run_verosad*.py`
  scripts.
- **Confirmed, not just assumed:** the fork's `state.copy()` + in-place
  `veros_routine` step does compose through
  `jax.grad(jax.jit(...))` around `lax.scan(jax.checkpoint(...))` nested
  inside vercor's own `Coupler.run` — the standalone spike this doc
  originally called for turned out to be `run_verosad_grad.py` itself.
