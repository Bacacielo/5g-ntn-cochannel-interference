import numpy as np

from ntn_interference.link_budget import (
    dbm_to_mw,
    free_space_path_loss_db,
    mw_to_dbm,
    received_power_dbm,
    sinr_db,
    thermal_noise_power_dbm,
)


def test_dbm_mw_round_trip():
    powers_dbm = np.array([-100.0, -30.0, 0.0, 10.0])

    assert np.allclose(mw_to_dbm(dbm_to_mw(powers_dbm)), powers_dbm)


def test_free_space_path_loss_near_known_value():
    # At 2 GHz and 1 km, FSPL is approximately 98.47 dB.
    fspl = free_space_path_loss_db(1_000, 2.0e9)

    assert np.isclose(fspl, 98.47, atol=0.1)


def test_noise_power_for_5_mhz_and_7_db_nf():
    noise_dbm = thermal_noise_power_dbm(5.0e6, 7)

    assert np.isclose(noise_dbm, -100.01, atol=0.1)


def test_received_power_decreases_with_distance():
    near = received_power_dbm(60, 600_000, 2.0e9, 0, 2)
    far = received_power_dbm(60, 1_200_000, 2.0e9, 0, 2)

    assert far < near


def test_sinr_drops_with_more_interference():
    low_interference = sinr_db(signal_dbm=-95, interference_dbm=-120, noise_dbm=-100)
    high_interference = sinr_db(signal_dbm=-95, interference_dbm=-90, noise_dbm=-100)

    assert high_interference < low_interference

