"""Link-budget and RF power utilities."""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT_MPS = 299_792_458.0
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0


def dbm_to_mw(power_dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert dBm to milliwatts."""
    return 10 ** (np.asarray(power_dbm) / 10.0)


def mw_to_dbm(power_mw: float | np.ndarray) -> float | np.ndarray:
    """Convert milliwatts to dBm."""
    power_mw = np.asarray(power_mw)
    return 10.0 * np.log10(np.maximum(power_mw, np.finfo(float).tiny))


def db_to_linear(value_db: float | np.ndarray) -> float | np.ndarray:
    """Convert a dB ratio to a linear ratio."""
    return 10 ** (np.asarray(value_db) / 10.0)


def linear_to_db(value_linear: float | np.ndarray) -> float | np.ndarray:
    """Convert a linear ratio to dB."""
    value_linear = np.asarray(value_linear)
    return 10.0 * np.log10(np.maximum(value_linear, np.finfo(float).tiny))


def free_space_path_loss_db(distance_m: float | np.ndarray, frequency_hz: float) -> float | np.ndarray:
    """Compute free-space path loss in dB."""
    distance_m = np.asarray(distance_m)
    if np.any(distance_m <= 0):
        raise ValueError("distance_m must be positive")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    wavelength_m = SPEED_OF_LIGHT_MPS / frequency_hz
    return 20.0 * np.log10((4.0 * np.pi * distance_m) / wavelength_m)


def received_power_dbm(
    eirp_dbm: float,
    distance_m: float | np.ndarray,
    frequency_hz: float,
    receiver_gain_dbi: float,
    misc_losses_db: float,
    shadowing_db: float | np.ndarray = 0.0,
    interference_suppression_db: float = 0.0,
) -> float | np.ndarray:
    """Compute received power in dBm from EIRP and link losses."""
    fspl_db = free_space_path_loss_db(distance_m, frequency_hz)
    return (
        eirp_dbm
        - fspl_db
        - misc_losses_db
        - shadowing_db
        - interference_suppression_db
        + receiver_gain_dbi
    )


def thermal_noise_power_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Compute receiver noise power in dBm."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return THERMAL_NOISE_DENSITY_DBM_PER_HZ + 10.0 * np.log10(bandwidth_hz) + noise_figure_db


def sinr_db(signal_dbm: float | np.ndarray, interference_dbm: float | np.ndarray, noise_dbm: float) -> float | np.ndarray:
    """Compute SINR in dB from signal, aggregate interference, and noise powers."""
    signal_mw = dbm_to_mw(signal_dbm)
    interference_mw = dbm_to_mw(interference_dbm)
    noise_mw = dbm_to_mw(noise_dbm)
    return linear_to_db(signal_mw / (interference_mw + noise_mw))

