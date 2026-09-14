"""The coupled model differentiated by the calibration.

Veros ocean + JCM land + JAX-GCM atmosphere, coupled through VerCOR on the global
4 degree grid, built as in `vercor/examples/run_jcm_with_veros.py`.

The rollout is split into segments so that the last `average_days` daily states
are observable: `Coupler.run` returns only its final state, so a 10-day rollout
whose last 3 states are needed becomes the chain [8, 1, 1]. Chaining is function
composition, so reverse-mode AD still differentiates the whole rollout in one pass.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


def _ensure_vercor_importable() -> None:
    """Use the vercor checkout in this repository when vercor is not installed."""
    try:
        import vercor  # noqa: F401
    except ModuleNotFoundError:
        checkout = Path(__file__).resolve().parents[4] / "vercor"
        if not checkout.is_dir():
            raise
        sys.path[:0] = [str(checkout), str(checkout / "veros")]


_ensure_vercor_importable()

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veros_gcm,
)
from vercor.setups._external.veros_state import set_veros_variable
from vercor.topology import SurfaceMaskPolicy

START_DATE = datetime(2000, 1, 3, 0, 0, 0)
DT_SECONDS = 86400.0
RUN_ORDER = ["OCN", "LND", "ATM"]


@dataclass(frozen=True)
class Grid:
    """Static grid information the observables need."""

    zt: np.ndarray           # (nz,) T-point depths, negative down, surface last
    mask: np.ndarray         # (nx, ny, nz) wet-cell mask, halo removed
    longitude: np.ndarray
    latitude: np.ndarray

    @property
    def surface(self) -> int:
        return self.mask.shape[2] - 1


def _components(restore_to_climatology: bool):
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=True,
            restore_to_climatology=restore_to_climatology,
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
    return ocn, jcm_setup.land, jcm_setup.atmosphere


def _exchanges():
    return (
        Exchange(source="ATM", target="OCN", fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
                 regridder_factory=bilinear),
        Exchange(source="OCN", target="ATM", fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                 regridder_factory=bilinear),
        Exchange(source="LND", target="ATM", fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                 regridder_factory=bilinear),
        Exchange(source="ATM", target="LND", fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                 regridder_factory=bilinear),
    )


def segment_lengths(days: int, average_days: int) -> list[int]:
    """Segment the rollout so each of the last `average_days` states ends a segment."""
    if not 1 <= average_days <= days:
        raise ValueError(f"average_days must be in [1, {days}], got {average_days}")
    return [days - average_days + 1] + [1] * (average_days - 1)


class CoupledModel:
    """A `days`-long coupled rollout as a differentiable function of (c_k, c_eps).

    With `restore_to_climatology` the ocean's surface heat flux carries a term
    `qnec * (sst_clim - SST)` that relaxes sea surface temperature toward observed
    climatology, with `qnec = -dQ/dSST` from the bulk formulae (order 10 W/m^2/K, so
    order 100 days for a 50 m layer). The atmosphere-ocean exchange is unaffected
    either way. The vercor gradient examples enable it; the VerosConfig default is off.
    """

    def __init__(self, days: int, average_days: int, restore_to_climatology: bool = True):
        self.days = days
        self.average_days = average_days
        self.restore_to_climatology = restore_to_climatology

        ocn, lnd, atm = _components(restore_to_climatology)
        exchanges = _exchanges()

        def make_coupler(start: datetime, steps: int) -> Coupler:
            return Coupler(
                clock=Clock(start=start, dt_seconds=DT_SECONDS, steps=steps, calendar="noleap"),
                components=[ocn, lnd, atm],
                exchanges=exchanges,
                run_order=RUN_ORDER,
                runtime=RuntimeOptions(topology=SurfaceMaskPolicy(),
                                       dtype=DTypePolicy(enable_x64=True)),
                log_level="WARNING",  # per-step INFO logging fires a host callback per step
            )

        self.couplers, offset = [], 0
        for steps in segment_lengths(days, average_days):
            self.couplers.append(make_coupler(START_DATE + timedelta(seconds=offset * DT_SECONDS), steps))
            offset += steps

        self.initial_state = self.couplers[0].initial_state()
        for coupler in self.couplers[1:]:
            coupler.initial_state()  # force preparation before the run

        variables = self.initial_state._component_state("OCN").payload.variables
        self.grid = Grid(
            zt=np.asarray(variables.zt),
            mask=np.asarray(variables.maskT[2:-2, 2:-2, :]).astype(bool),
            longitude=np.asarray(ocn.grid.longitude),
            latitude=np.asarray(ocn.grid.latitude),
        )

    def ocean_states(self, c_k, c_eps) -> list:
        """Ocean payloads at the end of every segment, i.e. the last daily states."""
        state = self.initial_state
        states = []
        for coupler in self.couplers:
            component = state._component_state("OCN")
            payload = set_veros_variable(component.payload, "c_k", c_k)
            payload = set_veros_variable(payload, "c_eps", c_eps)
            state = coupler.run(state._with_component_state("OCN", component.with_payload(payload)),
                                output=None)
            states.append(state._component_state("OCN").payload)
        return states
