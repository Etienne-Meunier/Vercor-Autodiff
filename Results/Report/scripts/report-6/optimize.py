"""Report 6: a real optimization run using the teacher-forced windowed
method validated in Report 5 (debug_script/19), with the one known rough
edge fixed -- a fixed Adam learning rate made c_k overshoot its optimum
and never come back (Report 5: c_k reached 0.1386 vs true 0.1, still
climbing at the last window). Fix: cosine-decay the learning rate across
windows, so early windows move fast (letting the much-slower c_eps catch
up) while late windows refine gently (letting c_k settle near its optimum
instead of sailing past it).

W=10 steps/window (within the trustworthy horizon per Report 3/4),
N_WINDOWS=25 (250 total simulated days -- longer than Report 5's 150, to
give c_eps more total updates), teacher forcing (every window starts from
the true trajectory's state, per Report 5) so cost stays linear in
N_WINDOWS with no chaos wall.
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

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup, VerosConfig, make_jcm_land_atmosphere, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, ATMOSPHERE_TO_VEROS_FORCING_FIELDS, JCM_LAND_TO_ATMOSPHERE_FIELDS, OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-6")
os.makedirs(OUT_DIR, exist_ok=True)

W = 10
N_WINDOWS = 25            # 250 total simulated days
PEAK_LR = 3e-2
FINAL_LR = 2e-3           # cosine decay from PEAK_LR to FINAL_LR over N_WINDOWS

TRUE_C_K, TRUE_C_EPS = 0.1, 0.7
INIT_C_K, INIT_C_EPS = 0.05, 0.4

ocn = make_veros_gcm(config=VerosConfig(setup="global_4deg_learning", uses_atmosphere_forcing=True, restore_to_climatology=True, jitted=True, execution="jax"))
jcm_setup = make_jcm_land_atmosphere(ocn.grid, config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)))
lnd = jcm_setup.land
atm = jcm_setup.atmosphere
exchanges = (
    Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
    Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
    Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
    Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
)


def make_window_coupler(start_date, n_steps):
    clock = Clock(start=start_date, dt_seconds=86400.0, steps=n_steps, calendar="noleap")
    return Coupler(clock=clock, components=[ocn, lnd, atm], exchanges=exchanges, run_order=["OCN", "LND", "ATM"],
                    runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTypePolicy(enable_x64=True)))


window_starts = [datetime(2000, 1, 3) + timedelta(days=i * W) for i in range(N_WINDOWS)]
window_couplers = [make_window_coupler(s, W) for s in window_starts]
for cpl in window_couplers:
    cpl.initial_state()

base_state = window_couplers[0].initial_state()


def rollout_window(cpl, state, c_k, c_eps):
    component_state = state._component_state("OCN")
    payload = set_veros_variable(component_state.payload, "c_k", c_k)
    payload = set_veros_variable(payload, "c_eps", c_eps)
    state2 = state._with_component_state("OCN", component_state.with_payload(payload))
    return cpl.run(state2, output=None)


print(f"Generating {N_WINDOWS}-window true trajectory (W={W} steps/window, true c_k={TRUE_C_K}, c_eps={TRUE_C_EPS}) ...")
true_states = [base_state]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()
    next_state = jax.jit(lambda s, cpl=cpl: rollout_window(cpl, s, jnp.asarray(TRUE_C_K), jnp.asarray(TRUE_C_EPS)))(true_states[-1])
    true_states.append(jax.lax.stop_gradient(next_state))
    print(f"  window {i} target done ({time.time()-t0:.1f}s)")
target_temps = [s._component_state("OCN").payload.variables.temp for s in true_states[1:]]

lr_schedule = optax.cosine_decay_schedule(init_value=PEAK_LR, decay_steps=N_WINDOWS, alpha=FINAL_LR / PEAK_LR)
params = {"c_k": jnp.asarray(INIT_C_K), "c_eps": jnp.asarray(INIT_C_EPS)}
optimizer = optax.adam(lr_schedule)
opt_state = optimizer.init(params)

print(f"\n{'window':>6s}  {'lr':>8s}  {'loss':>14s}  {'c_k':>10s}  {'c_eps':>10s}")
print(f"{'init':>6s}  {'':>8s}  {'':>14s}  {INIT_C_K:10.4f}  {INIT_C_EPS:10.4f}  (true: {TRUE_C_K}, {TRUE_C_EPS})")

history = [(INIT_C_K, INIT_C_EPS, None)]
for i, cpl in enumerate(window_couplers):
    t0 = time.time()
    start_state = true_states[i]

    def loss_fn(p, cpl=cpl, start_state=start_state, target=target_temps[i]):
        result = rollout_window(cpl, start_state, p["c_k"], p["c_eps"])
        return jnp.sum((result._component_state("OCN").payload.variables.temp - target) ** 2)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    ck_val, ceps_val = float(params["c_k"]), float(params["c_eps"])
    history.append((ck_val, ceps_val, float(loss_val)))
    print(f"{i:6d}  {float(lr_schedule(i)):8.4f}  {float(loss_val):14.6e}  {ck_val:10.4f}  {ceps_val:10.4f}  ({time.time()-t0:.1f}s)")

FINAL_C_K, FINAL_C_EPS = history[-1][0], history[-1][1]
print("\nrecovered c_k:  ", FINAL_C_K, " (true:", TRUE_C_K, ")")
print("recovered c_eps:", FINAL_C_EPS, " (true:", TRUE_C_EPS, ")")
print(f"\ntotal simulated span covered by targets: {N_WINDOWS * W} days, each gradient call spanned only {W} steps")

# --- figure: trajectory ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

windows = list(range(len(history) - 1))
losses = [h[2] for h in history[1:]]
axs[0].plot(windows, losses, "o-", color="tab:purple")
axs[0].set_xlabel("window")
axs[0].set_ylabel("loss (sum squared temp error)")
axs[0].set_yscale("log")
axs[0].set_title("Loss vs window (log scale)")
axs[0].grid(alpha=0.3)

cks = [h[0] for h in history]
ceps = [h[1] for h in history]
axs[1].plot(cks, ceps, "-o", color="tab:purple", markersize=4, label="optimization path")
axs[1].plot(cks[0], ceps[0], "o", color="black", markersize=8, label="start")
axs[1].plot(cks[-1], ceps[-1], "s", color="tab:purple", markersize=8, label="final")
axs[1].plot(TRUE_C_K, TRUE_C_EPS, "*", color="red", markersize=20, markeredgecolor="black", label="true", zorder=5)
axs[1].set_xlabel("c_k")
axs[1].set_ylabel("c_eps")
axs[1].set_title("Parameter trajectory")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 6: teacher-forced windowed calibration, {N_WINDOWS} windows x {W} steps ({N_WINDOWS*W} simulated days), cosine LR decay")
fig.tight_layout()
traj_path = os.path.join(OUT_DIR, "trajectory.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"saved {traj_path}")
