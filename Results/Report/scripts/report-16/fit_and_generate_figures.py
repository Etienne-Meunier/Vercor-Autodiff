"""Report 16: c_k/c_eps fit at N_STEPS=20, mixed-layer-depth (MLD) loss.

Same recipe as report-15 (single value_and_grad call over the full
rollout, optax.adam + cosine-decay lr, N_ITERS=30, N_STEPS=20, start
c_k=0.05/c_eps=0.4) EXCEPT the observable: the loss is now the sum of
squared error on the **mixed layer depth diagnosed from the final state**
-- not on temperature.

MLD is a diagnostic of the density profile: the depth at which potential
density first exceeds `prho(reference level) + 0.03 kg/m^3`, linearly
interpolated between the two levels that bracket that crossing. It is
exactly the quantity `c_k`/`c_eps` (TKE mixing length / dissipation) act
on, so it should carry more signal per cell than surface temperature,
which only sees the mixing indirectly through the heat it redistributes.

The MLD kernel here is the `get_index_mld` / `mld_from_index` split from
the Veros-Autodiff checkout's `setups/global_4deg/global_4deg_mld_learning.py`
(gradient-validated in that repo's `report-mld-2`), ported to operate on
the final coupled-run payload instead of being bolted into `after_timestep`.
The split matters for differentiability:

 - `get_index_mld` does the *discrete* level selection (argmax/argmin over
   boolean comparisons on prho). Index selection carries no gradient by
   construction, so no NaN/sentinel value ever reaches a differentiated
   primitive.
 - `mld_from_index` is a plain gather + arithmetic on those indices, fully
   differentiable, with the one division guarded by `well_defined` *before*
   dividing (0/0 at degenerate columns would otherwise produce a NaN
   gradient that contaminates every other column once summed into one loss).

Unlike Veros-Autodiff's `report-mld-2`, this uses the **instantaneous** MLD
of the last state -- no `mld_ma` moving average, no circular buffer.

Not every column has a well-defined MLD: land columns, and (with this
setup's coarse nz=15 grid) columns where no level in the column is denser
than the surface reference + 0.03 -- deep-mixed high-latitude water. Those
are excluded from the loss via the same `well_defined` mask the diagnostic
already computes, intersected with the target run's mask so the loss
domain is constant across iterations.

Script: `Results/Report/scripts/report-16/fit_and_generate_figures.py`.
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veros_gcm,
)
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

# Env overrides exist only for the smoke test (2 steps / 2 iters into a scratch
# directory) that checks the MLD kernel wiring before committing to a full run.
OUT_DIR = os.environ.get("REPORT16_OUT_DIR", os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-16"))
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = int(os.environ.get("REPORT16_N_STEPS", 20))
N_ITERS = int(os.environ.get("REPORT16_N_ITERS", 30))
LEARNING_RATE = 2e-2  # peak lr of the cosine-decay schedule

MLD_REFERENCE_DEPTH = -10.0  # m, negative down -- same as global_4deg_mld_learning.py
MLD_REFERENCE_OFFSET = 0.03  # kg/m^3 density offset defining the mixed layer base

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4


# --- MLD diagnostic (ported from Veros-Autodiff global_4deg_mld_learning.py) ---

def get_index_mld(prho, maskT, zt, reference_depth, reference_offset=MLD_REFERENCE_OFFSET):
    """Level indices bracketing the MLD -- discrete selection only, no gradient path.

    ridx (reference level) is recomputed from zt, never sliced to a dynamic length,
    so this stays jit-safe. i_below/i_above come out of argmax/argmin gated by
    boolean comparisons on prho -- comparisons carry no gradient, so these are
    naturally zero-gradient w.r.t. prho. well_defined marks columns with no valid
    below/above level (land, or a column nowhere denser than the reference).
    Land excluded via maskT (authoritative regardless of eq_of_state_type), not
    isnan(prho) -- under nonlin2 land cells are exactly 0.0, not NaN.
    """
    level = jnp.arange(zt.shape[-1])
    ridx = jnp.max(jnp.where(zt < reference_depth, level, -1))

    prho_reference = prho[:, :, ridx] + reference_offset
    valid = maskT.astype(bool) & (level <= ridx)

    drho = prho - prho_reference[:, :, jnp.newaxis]

    below_mask = valid & (drho > 0)
    has_below = jnp.any(below_mask, axis=-1)
    i_below = jnp.argmax(jnp.where(below_mask, zt, -jnp.inf), axis=-1)
    depth_below = zt[i_below]

    above_mask = valid & (drho < 0) & (zt > depth_below[:, :, jnp.newaxis])
    has_above = jnp.any(above_mask, axis=-1)
    i_above = jnp.argmin(jnp.where(above_mask, zt, jnp.inf), axis=-1)

    well_defined = has_below & has_above
    return ridx, i_below, i_above, well_defined


def mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset=MLD_REFERENCE_OFFSET):
    """MLD from precomputed level indices -- plain gather + arithmetic, differentiable.

    Returns a finite value everywhere (0.0 at degenerate columns); callers mask with
    `well_defined`. Returning NaN there instead would be fine for plotting but poisons
    the gradient of any sum over the field, so the masking is left to the caller.
    """
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below = jnp.take_along_axis(prho, i_below[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    prho_above = jnp.take_along_axis(prho, i_above[:, :, jnp.newaxis], axis=-1)[:, :, 0]
    depth_below = zt[i_below]
    depth_above = zt[i_above]

    # denom is 0/0 at degenerate columns -- guard with a safe placeholder BEFORE
    # dividing, mask the result afterwards, or the NaN gradient contaminates the
    # whole summed loss.
    denom = jnp.where(well_defined, prho_above - prho_below, 1.0)
    mld = (prho_reference - prho_below) / denom * (depth_above - depth_below) + depth_below
    return jnp.where(well_defined, mld, 0.0)


ocn = make_veros_gcm(
    config=VerosConfig(
        setup="global_4deg_learning",
        uses_atmosphere_forcing=True,
        restore_to_climatology=True,
        jitted=True,
        execution="jax",
    ),
)
jcm_setup = make_jcm_land_atmosphere(
    ocn.grid,
    config=JCMLandAtmosphereConfig(
        atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True),
    ),
)
lnd = jcm_setup.land
atm = jcm_setup.atmosphere

clock = Clock(start=datetime(2000, 1, 3, 0, 0, 0), dt_seconds=86400.0, steps=N_STEPS, calendar="noleap")
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)
cpl = Coupler(
    clock=clock,
    components=[ocn, lnd, atm],
    exchanges=exchanges,
    run_order=["OCN", "LND", "ATM"],
    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)),
)
initial_state = cpl.initial_state()

_init_vars = initial_state._component_state("OCN").payload.variables
ZT = jnp.asarray(_init_vars.zt)
MASKT = jnp.asarray(_init_vars.maskT[2:-2, 2:-2, :]).astype(bool)
_zt_np = np.asarray(_init_vars.zt)
_ridx = int(np.max(np.where(_zt_np < MLD_REFERENCE_DEPTH, np.arange(_zt_np.size), -1)))
print(f"zt (m, cell centers): {np.round(_zt_np, 1).tolist()}", flush=True)
print(f"MLD reference level index: {_ridx} (zt={_zt_np[_ridx]:.1f} m), offset {MLD_REFERENCE_OFFSET} kg/m^3", flush=True)


def final_ocn_payload(c_k, c_eps):
    component_state = initial_state._component_state("OCN")
    payload = component_state.payload
    payload = set_veros_variable(payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state = initial_state._with_component_state("OCN", component_state.with_payload(payload))
    result = cpl.run(state, output=None)
    return result._component_state("OCN").payload


def mld_of_payload(payload):
    """Instantaneous MLD of the final state -- no moving average, last state only."""
    prho = payload.variables.prho[2:-2, 2:-2, :]
    ridx, i_below, i_above, well_defined = get_index_mld(prho, MASKT, ZT, MLD_REFERENCE_DEPTH)
    mld = mld_from_index(prho, ZT, ridx, i_below, i_above, well_defined)
    return mld, well_defined


def final_ocn_mld(c_k, c_eps):
    return mld_of_payload(final_ocn_payload(c_k, c_eps))


print(f"Generating synthetic target with true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS} (N_STEPS={N_STEPS}) ...", flush=True)
target_mld, target_well_defined = jax.jit(final_ocn_mld)(jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS))
target_mld = jax.lax.stop_gradient(target_mld)

# Loss domain: columns with a well-defined MLD in the TARGET run, held fixed across
# iterations (so the loss is comparable iteration to iteration). Columns that lose
# their definition in a candidate run are additionally dropped inside loss_fn --
# counted below to confirm that stays a no-op.
loss_mask = jnp.asarray(np.asarray(target_well_defined))
_loss_mask_np = np.asarray(target_well_defined)
N_MLD_CELLS = int(_loss_mask_np.sum())
N_OCEAN_CELLS = int(np.asarray(MASKT)[:, :, -1].sum())
print(f"columns with well-defined MLD in loss: {N_MLD_CELLS} / {N_OCEAN_CELLS} surface ocean columns "
      f"({_loss_mask_np.size} total surface cells)", flush=True)
_tgt = np.asarray(target_mld)[_loss_mask_np]
print(f"target MLD range: {_tgt.min():.1f} to {_tgt.max():.1f} m (mean {_tgt.mean():.1f} m)", flush=True)


def loss_fn(params):
    mld, well_defined = final_ocn_mld(params["c_k"], params["c_eps"])
    diff = jnp.where(loss_mask & well_defined, mld - target_mld, 0.0)
    return jnp.sum(diff ** 2)


TRAJ_PATH = os.path.join(OUT_DIR, "trajectory.npy")

if os.path.exists(TRAJ_PATH):
    trajectory = [tuple(row) for row in np.load(TRAJ_PATH).tolist()]
    print(f"resuming optimization from checkpoint ({len(trajectory) - 1} / {N_ITERS} iters already done)", flush=True)
else:
    trajectory = [(INIT_C_K, INIT_C_EPS, None)]

params = {"c_k": jnp.asarray(trajectory[-1][0]), "c_eps": jnp.asarray(trajectory[-1][1])}
schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=N_ITERS, alpha=0.0)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)
value_and_grad_fn = jax.value_and_grad(loss_fn)

# Checkpoint row k+1 holds (params AFTER iteration k's update, loss evaluated
# BEFORE it, i.e. at row k's params). So the params that produced the loss stored
# in row k+1 are row k's -- pair them that way when picking the best iterate, or
# the reported "best" parameters are one adam step past the ones that were scored.
def best_iterate(traj):
    best_i, best_l = -1, float("inf")
    best_p = (traj[0][0], traj[0][1])
    for k in range(len(traj) - 1):
        loss = traj[k + 1][2]
        if loss is not None and not np.isnan(loss) and loss < best_l:
            best_l, best_i = loss, k
            best_p = (traj[k][0], traj[k][1])
    return best_i, best_l, best_p[0], best_p[1]


best_iter, best_loss, best_ck, best_ceps = best_iterate(trajectory)
iter_wall_times = []

print(f"\n{'iter':>4s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}  {'dL/dc_k':>12s}  {'dL/dc_eps':>12s}  {'lr':>10s}  {'wall_s':>8s}", flush=True)
print(f"{'init':>4s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})", flush=True)

t_start = time.time()
for iteration in range(len(trajectory) - 1, N_ITERS):
    t0 = time.time()
    prev_ck, prev_ceps = float(params["c_k"]), float(params["c_eps"])
    loss_value, grads = value_and_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    dt = time.time() - t0
    iter_wall_times.append(dt)

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    loss_val = float(loss_value)
    trajectory.append((ck_val, ceps_val, loss_val))
    print(f"{iteration:4d}  {loss_val:14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  "
          f"{float(grads['c_k']):12.4e}  {float(grads['c_eps']):12.4e}  "
          f"{float(schedule(iteration)):10.4f}  {dt:8.1f}", flush=True)
    np.save(TRAJ_PATH, np.array([[t[0], t[1], t[2] if t[2] is not None else np.nan] for t in trajectory]))

    if loss_val < best_loss:
        best_loss, best_iter = loss_val, iteration
        best_ck, best_ceps = prev_ck, prev_ceps  # params the loss was evaluated at

total_wall = time.time() - t_start
FINAL_C_K, FINAL_C_EPS = trajectory[-1][0], trajectory[-1][1]
FINAL_LOSS = trajectory[-1][2]
print(f"\ntotal descent wall time: {total_wall:.1f}s over {len(iter_wall_times)} iterations "
      f"(mean {np.mean(iter_wall_times):.1f}s/iter)" if iter_wall_times else "\nno new iterations run (already complete)", flush=True)
print("recovered c_k (final):  ", FINAL_C_K, " (true:", TRUE_C_K, ")", flush=True)
print("recovered c_eps (final):", FINAL_C_EPS, " (true:", TRUE_C_EPS, ")", flush=True)
print(f"best iteration: {best_iter}, loss={best_loss:.6e} at c_k={best_ck:.4f}, c_eps={best_ceps:.4f} "
      f"(the params that loss was evaluated at, i.e. before iteration {best_iter}'s update)", flush=True)

# --- snapshot rollouts: init, best, final (target already computed above) ---
print("\nrolling out initial guess for snapshot ...", flush=True)
init_mld, init_wd = jax.jit(final_ocn_mld)(jnp.asarray(INIT_C_K), jnp.asarray(INIT_C_EPS))
print("rolling out best-iteration params for snapshot ...", flush=True)
best_mld, best_wd = jax.jit(final_ocn_mld)(jnp.asarray(best_ck), jnp.asarray(best_ceps))
print("rolling out final params for snapshot ...", flush=True)
final_mld, final_wd = jax.jit(final_ocn_mld)(jnp.asarray(FINAL_C_K), jnp.asarray(FINAL_C_EPS))

for name, wd in (("init", init_wd), ("best", best_wd), ("final", final_wd)):
    n_lost = int((_loss_mask_np & ~np.asarray(wd)).sum())
    print(f"columns in the target MLD mask that lost their definition at {name}: {n_lost}", flush=True)

xt = np.asarray(ocn.grid.longitude)
yt = np.asarray(ocn.grid.latitude)


def masked(field, wd):
    return np.where(_loss_mask_np & np.asarray(wd), np.asarray(field), np.nan)


target_field = masked(target_mld, target_well_defined)
bias_initial = masked(init_mld, init_wd) - target_field
bias_best = masked(best_mld, best_wd) - target_field
bias_final = masked(final_mld, final_wd) - target_field

bias_max = np.nanmax(np.abs(np.concatenate([
    bias_initial[~np.isnan(bias_initial)],
    bias_best[~np.isnan(bias_best)],
    bias_final[~np.isnan(bias_final)],
])))

np.save(os.path.join(OUT_DIR, "target_mld.npy"), target_field)
np.save(os.path.join(OUT_DIR, "bias_initial.npy"), bias_initial)
np.save(os.path.join(OUT_DIR, "bias_best.npy"), bias_best)
np.save(os.path.join(OUT_DIR, "bias_final.npy"), bias_final)

# --- figure 1: trajectory ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

iters = list(range(len(trajectory) - 1))
losses = [t[2] for t in trajectory[1:]]
axs[0].plot(iters, losses, "o-", color="tab:purple")
axs[0].axvline(best_iter, color="tab:orange", linestyle="--", alpha=0.7, label=f"best (iter {best_iter})")
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("loss (sum squared MLD error, m^2)")
axs[0].set_yscale("log")
axs[0].set_title("Loss vs iteration (log scale)")
axs[0].legend()
axs[0].grid(alpha=0.3)

cks = [t[0] for t in trajectory]
ceps = [t[1] for t in trajectory]
axs[1].plot(cks, ceps, "-o", color="tab:purple", markersize=4, label="GD trajectory")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(best_ck, best_ceps, "D", color="tab:orange", markersize=9, label="best", zorder=4)
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:purple", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 16: c_k/c_eps recovery, {N_STEPS}-step coupled rollout, "
             f"MLD (last state) loss, adam + cosine-decay lr")
fig.tight_layout()
traj_path = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"saved {traj_path}", flush=True)

# --- figure 2: MLD snapshot, three fitted parameter states ---
fig, axs = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="viridis", shading="auto")
axs[0].set_title(f"target MLD (c_k={TRUE_C_K}, c_eps={TRUE_C_EPS})")
fig.colorbar(im0, ax=axs[0], label="MLD (m)", shrink=0.8)

axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title(f"init - target (c_k={INIT_C_K}, c_eps={INIT_C_EPS})")

axs[2].pcolormesh(xt, yt, bias_best.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title(f"best (loss at iter {best_iter}) - target\n(c_k={best_ck:.4f}, c_eps={best_ceps:.4f})")

im3 = axs[3].pcolormesh(xt, yt, bias_final.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[3].set_title(f"final - target\n(c_k={FINAL_C_K:.4f}, c_eps={FINAL_C_EPS:.4f})")
fig.colorbar(im3, ax=[axs[1], axs[2], axs[3]], label="MLD bias (m)", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"Report 16: mixed-layer-depth bias after {N_STEPS} steps, MLD-loss recipe")
fig.subplots_adjust(top=0.82)

snap_path = os.path.join(OUT_DIR, "mld_snapshot.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"saved {snap_path}", flush=True)
