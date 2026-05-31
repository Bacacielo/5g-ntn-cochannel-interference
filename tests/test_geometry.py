import numpy as np

from ntn_interference.geometry import (
    central_angle_from_elevation_deg,
    elevation_from_central_angle_deg,
    random_interferer_elevations_deg,
    slant_range_m,
)


def test_slant_range_decreases_as_elevation_increases():
    low_elevation = slant_range_m(10, 600_000)
    high_elevation = slant_range_m(60, 600_000)

    assert high_elevation < low_elevation
    assert high_elevation > 0


def test_random_interferer_elevations_shape_and_bounds():
    rng = np.random.default_rng(42)
    values = random_interferer_elevations_deg(rng, count=4, trials=100, min_deg=10, max_deg=60)

    assert values.shape == (100, 4)
    assert np.all(values >= 10)
    assert np.all(values <= 60)


def test_central_angle_elevation_round_trip():
    elevations = np.array([10.0, 30.0, 60.0])
    central_angles = central_angle_from_elevation_deg(elevations, 600_000)
    recovered = elevation_from_central_angle_deg(central_angles, 600_000)

    assert np.allclose(recovered, elevations)


def test_visible_shell_sampling_favors_lower_elevations():
    rng = np.random.default_rng(42)
    values = random_interferer_elevations_deg(rng, count=1, trials=10_000, min_deg=10, max_deg=60)

    assert float(np.median(values)) < 35.0
