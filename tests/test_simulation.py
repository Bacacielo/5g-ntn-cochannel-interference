import numpy as np

from ntn_interference.config import SimulationConfig
from ntn_interference.simulation import Scenario, simulate_scenario


def test_simulate_scenario_accepts_single_eirp_value():
    cfg = SimulationConfig(
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=5.0e6,
        altitude_m=600_000,
        serving_elevation_deg=60,
        interferer_elevation_min_deg=10,
        interferer_elevation_max_deg=60,
        receiver_gain_dbi=0,
        receiver_noise_figure_db=7,
        misc_losses_db=2,
        serving_shadow_std_db=2,
        interferer_shadow_std_db=4,
        eirp_scenarios_dbm=[60],
        interferer_counts=[20],
        monte_carlo_trials=100,
        outage_thresholds_db=[0],
        random_seed=42,
    )

    summary, samples, powers = simulate_scenario(cfg, Scenario("baseline"), np.random.default_rng(42))

    assert len(summary) == 1
    assert summary.loc[0, "eirp_dbm"] == 60
    assert summary.loc[0, "interferer_count"] == 20
    assert "outage_lt_0_db" in summary.columns
    assert not samples.empty
    assert not powers.empty


def test_beam_discrimination_improves_mean_sinr():
    cfg = SimulationConfig(
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=5.0e6,
        altitude_m=600_000,
        serving_elevation_deg=60,
        interferer_elevation_min_deg=10,
        interferer_elevation_max_deg=60,
        receiver_gain_dbi=0,
        receiver_noise_figure_db=7,
        misc_losses_db=2,
        serving_shadow_std_db=2,
        interferer_shadow_std_db=4,
        eirp_scenarios_dbm=[60],
        interferer_counts=[20],
        monte_carlo_trials=100,
        outage_thresholds_db=[0],
        random_seed=42,
    )
    rng_seed = 123

    baseline, _, _ = simulate_scenario(cfg, Scenario("baseline"), np.random.default_rng(rng_seed))
    suppressed, _, _ = simulate_scenario(
        cfg,
        Scenario("beam_discrimination", beam_discrimination_db=10),
        np.random.default_rng(rng_seed),
    )

    assert suppressed.loc[0, "mean_sinr_db"] > baseline.loc[0, "mean_sinr_db"]


def test_custom_sample_interferer_count_is_returned():
    cfg = SimulationConfig(
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=5.0e6,
        altitude_m=600_000,
        serving_elevation_deg=60,
        interferer_elevation_min_deg=10,
        interferer_elevation_max_deg=60,
        receiver_gain_dbi=0,
        receiver_noise_figure_db=7,
        misc_losses_db=2,
        serving_shadow_std_db=2,
        interferer_shadow_std_db=4,
        eirp_scenarios_dbm=[60],
        interferer_counts=[7],
        monte_carlo_trials=100,
        outage_thresholds_db=[0],
        random_seed=42,
    )

    _, samples, _ = simulate_scenario(
        cfg,
        Scenario("interactive"),
        np.random.default_rng(42),
        sample_interferer_counts={7},
    )

    assert set(samples["interferer_count"]) == {7}
    assert len(samples) == 100
