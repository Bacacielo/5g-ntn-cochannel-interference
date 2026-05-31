"""Geometry helpers for simple LEO satellite link calculations."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def slant_range_m(
    elevation_deg: float | np.ndarray,
    altitude_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> float | np.ndarray:
    """Return satellite slant range in meters for a ground user.

    The model assumes a spherical Earth, a user at the Earth's surface, and a
    satellite at fixed altitude. Elevation is measured above the local horizon.
    """
    elevation_rad = np.deg2rad(elevation_deg)
    orbital_radius_m = earth_radius_m + altitude_m
    range_m = (
        np.sqrt(orbital_radius_m**2 - (earth_radius_m * np.cos(elevation_rad)) ** 2)
        - earth_radius_m * np.sin(elevation_rad)
    )
    return range_m


def central_angle_from_elevation_deg(
    elevation_deg: float | np.ndarray,
    altitude_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> float | np.ndarray:
    """Return Earth-center angle between UE and satellite subpoint in degrees."""
    elevation_rad = np.deg2rad(elevation_deg)
    orbital_radius_m = earth_radius_m + altitude_m
    range_m = slant_range_m(elevation_deg, altitude_m, earth_radius_m)
    sin_psi = range_m * np.cos(elevation_rad) / orbital_radius_m
    cos_psi = (earth_radius_m + range_m * np.sin(elevation_rad)) / orbital_radius_m
    return np.rad2deg(np.arctan2(sin_psi, cos_psi))


def elevation_from_central_angle_deg(
    central_angle_deg: float | np.ndarray,
    altitude_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> float | np.ndarray:
    """Return elevation angle in degrees from Earth-center geometry."""
    central_angle_rad = np.deg2rad(central_angle_deg)
    orbital_radius_m = earth_radius_m + altitude_m
    vertical_m = orbital_radius_m * np.cos(central_angle_rad) - earth_radius_m
    horizontal_m = orbital_radius_m * np.sin(central_angle_rad)
    return np.rad2deg(np.arctan2(vertical_m, horizontal_m))


def random_interferer_elevations_deg(
    rng: np.random.Generator,
    count: int,
    trials: int,
    min_deg: float,
    max_deg: float,
    altitude_m: float = 600_000.0,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Generate visible-shell random elevations with shape ``(trials, count)``.

    Satellites are sampled uniformly over the visible spherical-shell annulus
    defined by the elevation limits. This is more defensible than assigning a
    uniform probability directly to elevation angle.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if min_deg >= max_deg:
        raise ValueError("min_deg must be smaller than max_deg")
    if altitude_m <= 0:
        raise ValueError("altitude_m must be positive")
    if count == 0:
        return np.empty((trials, 0))
    psi_at_min_elevation_deg = central_angle_from_elevation_deg(min_deg, altitude_m, earth_radius_m)
    psi_at_max_elevation_deg = central_angle_from_elevation_deg(max_deg, altitude_m, earth_radius_m)
    cos_psi_min = np.cos(np.deg2rad(psi_at_min_elevation_deg))
    cos_psi_max = np.cos(np.deg2rad(psi_at_max_elevation_deg))
    cos_psi = rng.uniform(cos_psi_min, cos_psi_max, size=(trials, count))
    central_angle_deg = np.rad2deg(np.arccos(cos_psi))
    return elevation_from_central_angle_deg(central_angle_deg, altitude_m, earth_radius_m)
