"""Report 9: compare the atmospheric forcing applied to OCN under the two
configs used in debug_script/20 and the coupled reports -- live JCM
coupling (uses_atmosphere_forcing=True) vs prescribed climatological
forcing (uses_atmosphere_forcing=False, restore_to_climatology=True) --
over a longer forward-only rollout, to see directly (not just inferred
from gradient behavior) whether the two really drive the ocean with
qualitatively different dynamics: JCM's live, daily-evolving weather vs
a smooth, slowly-varying monthly climatology.

Forward-only, no gradients. Both configs are run day by day (single
Coupler rebuilt each day, matching the report-6/7/8 windowed pattern) so
the two forcing fields Veros itself computes and applies each step --
surface_taux/surface_tauy (wind stress) and forc_temp_surface (net heat
forcing) -- can be extracted at daily resolution.

Outputs: a GIF of the wind-stress-magnitude field side by side (JCM |
forced), and curve plots of domain-mean wind stress magnitude and
domain-mean forc_temp_surface vs day, both configs overlaid.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VERCOR_ROOT = os.path.join(_REPO_ROOT, "vercor")
sys.path.insert(0, os.path.join(_VERCOR_ROOT, "veros"))
sys.path.insert(0, _VERCOR_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

OUT_DIR = os.path.join(_REPO_ROOT, "Results", "Report", "figures", "report-9")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 365  # 1 year, daily resolution -- see the climatology cycle close and JCM's seasonal behavior
START = datetime(2001, 1, 3, 0, 0, 0)  # non-leap year: plain timedelta arithmetic never hits Feb 29
DTYPE = DTypePolicy(enable_x64=True)


def build_jcm_daily_couplers():
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
        config=JCMLandAtmosphereConfig(atmosphere=JAXGCMConfig(spinup=Spinup(enabled=False), jitted=True)),
    )
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere
    exchanges = (
        Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS, regridder_factory=bilinear),
        Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS, regridder_factory=bilinear),
        Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS, regridder_factory=bilinear),
        Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS, regridder_factory=bilinear),
    )

    def make_day(start_date):
        clock = Clock(start=start_date, dt_seconds=86400.0, steps=1, calendar="noleap")
        return Coupler(
            clock=clock,
            components=[ocn, lnd, atm],
            exchanges=exchanges,
            run_order=["OCN", "LND", "ATM"],
            runtime=RuntimeOptions(topology=SurfaceMaskPolicy(), dtype=DTYPE),
        )

    days = [make_day(START + timedelta(days=i)) for i in range(N_STEPS)]
    for cpl in days:
        cpl.initial_state()
    return days, ocn.grid


def build_forced_daily_couplers():
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=False,
            restore_to_climatology=True,
            jitted=True,
            execution="jax",
        ),
    )

    def make_day(start_date):
        clock = Clock(start=start_date, dt_seconds=86400.0, steps=1, calendar="noleap")
        return Coupler(clock=clock, components=[ocn], run_order=["OCN"], runtime=RuntimeOptions(dtype=DTYPE))

    days = [make_day(START + timedelta(days=i)) for i in range(N_STEPS)]
    for cpl in days:
        cpl.initial_state()
    return days, ocn.grid


def field_2d(payload, name, mask_name):
    val = np.asarray(getattr(payload.variables, name)[2:-2, 2:-2])
    mask = np.asarray(getattr(payload.variables, mask_name)[2:-2, 2:-2, -1]).astype(bool)
    return np.where(mask, val, np.nan)


def rollout_and_collect(couplers, label):
    print(f"\n[{label}] rolling out {N_STEPS} days ...", flush=True)
    state = couplers[0].initial_state()
    tau_mag_frames, forc_temp_frames = [], []
    tau_mag_mean, forc_temp_mean = [], []
    import time as _time

    for i, cpl in enumerate(couplers):
        t0 = _time.time()
        state = cpl.run(state, output=None)
        payload = state._component_state("OCN").payload
        taux = field_2d(payload, "surface_taux", "maskU")
        tauy = field_2d(payload, "surface_tauy", "maskV")
        tau_mag = np.sqrt(taux ** 2 + tauy ** 2)
        forc_temp = field_2d(payload, "forc_temp_surface", "maskT")

        tau_mag_frames.append(tau_mag.astype(np.float32))
        forc_temp_frames.append(forc_temp.astype(np.float32))
        tau_mag_mean.append(float(np.nanmean(tau_mag)))
        forc_temp_mean.append(float(np.nanmean(forc_temp)))
        print(f"  [{label}] day {i:3d}  tau_mag_mean={tau_mag_mean[-1]:.4e}  forc_temp_mean={forc_temp_mean[-1]:.4e}  ({_time.time() - t0:.1f}s)", flush=True)

    return {
        "tau_mag_frames": tau_mag_frames,
        "forc_temp_frames": forc_temp_frames,
        "tau_mag_mean": tau_mag_mean,
        "forc_temp_mean": forc_temp_mean,
    }


print("Building JCM (live atmosphere) daily couplers ...", flush=True)
jcm_days, grid = build_jcm_daily_couplers()
jcm_data = rollout_and_collect(jcm_days, "jcm")

print("\nBuilding forced (climatology) daily couplers ...", flush=True)
forced_days, _ = build_forced_daily_couplers()
forced_data = rollout_and_collect(forced_days, "forced")

np.savez(
    os.path.join(OUT_DIR, "forcing_comparison_data.npz"),
    jcm_tau_mag_mean=jcm_data["tau_mag_mean"],
    jcm_forc_temp_mean=jcm_data["forc_temp_mean"],
    forced_tau_mag_mean=forced_data["tau_mag_mean"],
    forced_forc_temp_mean=forced_data["forc_temp_mean"],
)

# --- curve figure ---
days = np.arange(N_STEPS)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(days, jcm_data["tau_mag_mean"], "-", color="tab:red", label="JCM (live atmosphere)")
axs[0].plot(days, forced_data["tau_mag_mean"], "-", color="tab:blue", label="forced (climatology)")
axs[0].set_ylabel("domain-mean |wind stress| (N/m^2)")
axs[0].set_title("Wind stress magnitude vs day")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(days, jcm_data["forc_temp_mean"], "-", color="tab:red", label="JCM (live atmosphere)")
axs[1].plot(days, forced_data["forc_temp_mean"], "-", color="tab:blue", label="forced (climatology)")
axs[1].set_ylabel("domain-mean forc_temp_surface (m degC/s)")
axs[1].set_xlabel("day")
axs[1].set_title("Surface heat forcing vs day")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(f"Report 9: atmospheric forcing, JCM vs climatology-forced, {N_STEPS}-day forward rollout")
fig.tight_layout()
curve_path = os.path.join(OUT_DIR, "forcing_curves.png")
fig.savefig(curve_path, dpi=150)
plt.close(fig)
print(f"\nsaved {curve_path}")

# --- GIF: side-by-side wind stress magnitude field ---
xt = np.asarray(grid.longitude)
yt = np.asarray(grid.latitude)

all_frames = jcm_data["tau_mag_frames"] + forced_data["tau_mag_frames"]
vmax = np.nanpercentile(np.concatenate([f.ravel() for f in all_frames]), 99)

GIF_STRIDE = 2 if N_STEPS > 180 else 1  # keep frame count/file size reasonable over a full year
gif_frames = []
for i in range(0, N_STEPS, GIF_STRIDE):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    im0 = axs[0].pcolormesh(xt, yt, jcm_data["tau_mag_frames"][i].T, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    axs[0].set_title("JCM (live atmosphere)")
    im1 = axs[1].pcolormesh(xt, yt, forced_data["tau_mag_frames"][i].T, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    axs[1].set_title("forced (climatology)")
    fig.colorbar(im1, ax=axs, label="|wind stress| (N/m^2)", shrink=0.8)
    fig.suptitle(f"Report 9: wind stress magnitude, day {i}")
    for ax in axs:
        ax.set_xlabel("xt")
    axs[0].set_ylabel("yt")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    gif_frames.append(Image.fromarray(buf).convert("RGB"))
    plt.close(fig)

gif_path = os.path.join(OUT_DIR, "forcing_evolution.gif")
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=120, loop=0)
print(f"saved {gif_path}")
