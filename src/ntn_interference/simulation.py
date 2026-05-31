"""Monte Carlo simulation CLI for 5G-NTN co-channel interference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ntn_interference.config import (
    MitigationConfig,
    SimulationConfig,
    load_mitigation_config,
    load_simulation_config,
)
from ntn_interference.geometry import random_interferer_elevations_deg, slant_range_m
from ntn_interference.link_budget import (
    db_to_linear,
    dbm_to_mw,
    linear_to_db,
    mw_to_dbm,
    received_power_dbm,
    sinr_db,
    thermal_noise_power_dbm,
)

SELECTED_CDF_COUNTS = {5, 20, 50}


@dataclass(frozen=True)
class Scenario:
    name: str
    activity_factor: float = 1.0
    beam_discrimination_db: float = 0.0


def _scenario_list(mitigation: MitigationConfig | None) -> list[Scenario]:
    if mitigation is None:
        return [Scenario("baseline")]
    return [
        Scenario("activity_reduction", activity_factor=mitigation.activity_factor),
        Scenario("beam_discrimination", beam_discrimination_db=mitigation.beam_discrimination_db),
    ]


def _aggregate_interference_dbm(
    cfg: SimulationConfig,
    scenario: Scenario,
    eirp_dbm: float,
    interferer_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    trials = cfg.monte_carlo_trials
    if interferer_count == 0:
        return np.full(trials, -300.0)

    active = rng.random((trials, interferer_count)) < scenario.activity_factor
    elevations_deg = random_interferer_elevations_deg(
        rng,
        interferer_count,
        trials,
        cfg.interferer_elevation_min_deg,
        cfg.interferer_elevation_max_deg,
        altitude_m=cfg.altitude_m,
    )
    ranges_m = slant_range_m(elevations_deg, cfg.altitude_m)
    shadowing_db = rng.normal(0.0, cfg.interferer_shadow_std_db, size=(trials, interferer_count))
    interferer_power_dbm = received_power_dbm(
        eirp_dbm=eirp_dbm,
        distance_m=ranges_m,
        frequency_hz=cfg.carrier_frequency_hz,
        receiver_gain_dbi=cfg.receiver_gain_dbi,
        misc_losses_db=cfg.misc_losses_db,
        shadowing_db=shadowing_db,
        interference_suppression_db=scenario.beam_discrimination_db,
    )
    interferer_power_mw = dbm_to_mw(interferer_power_dbm) * active
    total_interference_mw = interferer_power_mw.sum(axis=1)
    return mw_to_dbm(total_interference_mw)


def simulate_scenario(
    cfg: SimulationConfig,
    scenario: Scenario,
    rng: np.random.Generator,
    sample_interferer_counts: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one simulation scenario and return summary, SINR samples, and power rows."""
    summary_rows: list[dict[str, float | int | str]] = []
    sample_rows: list[pd.DataFrame] = []
    power_rows: list[dict[str, float | int | str]] = []
    noise_dbm = thermal_noise_power_dbm(cfg.bandwidth_hz, cfg.receiver_noise_figure_db)
    serving_range_m = slant_range_m(cfg.serving_elevation_deg, cfg.altitude_m)
    reference_eirp_dbm = cfg.eirp_scenarios_dbm[len(cfg.eirp_scenarios_dbm) // 2]
    sample_counts = SELECTED_CDF_COUNTS if sample_interferer_counts is None else sample_interferer_counts

    for eirp_dbm in cfg.eirp_scenarios_dbm:
        serving_shadow_db = rng.normal(0.0, cfg.serving_shadow_std_db, size=cfg.monte_carlo_trials)
        signal_dbm = received_power_dbm(
            eirp_dbm=eirp_dbm,
            distance_m=serving_range_m,
            frequency_hz=cfg.carrier_frequency_hz,
            receiver_gain_dbi=cfg.receiver_gain_dbi,
            misc_losses_db=cfg.misc_losses_db,
            shadowing_db=serving_shadow_db,
        )

        for interferer_count in cfg.interferer_counts:
            interference_dbm = _aggregate_interference_dbm(
                cfg=cfg,
                scenario=scenario,
                eirp_dbm=eirp_dbm,
                interferer_count=interferer_count,
                rng=rng,
            )
            sinr_values_db = sinr_db(signal_dbm, interference_dbm, noise_dbm)

            row: dict[str, float | int | str] = {
                "scenario": scenario.name,
                "eirp_dbm": eirp_dbm,
                "interferer_count": interferer_count,
                "mean_signal_dbm": float(np.mean(signal_dbm)),
                "mean_interference_dbm": float(mw_to_dbm(np.mean(dbm_to_mw(interference_dbm)))),
                "noise_dbm": float(noise_dbm),
                "mean_sinr_db": float(np.mean(sinr_values_db)),
                "mean_linear_sinr_db": float(linear_to_db(np.mean(db_to_linear(sinr_values_db)))),
                "median_sinr_db": float(np.median(sinr_values_db)),
                "p05_sinr_db": float(np.percentile(sinr_values_db, 5)),
                "p95_sinr_db": float(np.percentile(sinr_values_db, 95)),
            }
            for threshold_db in cfg.outage_thresholds_db:
                key = f"outage_lt_{threshold_db:g}_db"
                row[key] = float(np.mean(sinr_values_db < threshold_db))
            summary_rows.append(row)

            if interferer_count in sample_counts:
                sample_rows.append(
                    pd.DataFrame(
                        {
                            "scenario": scenario.name,
                            "eirp_dbm": eirp_dbm,
                            "interferer_count": interferer_count,
                            "sinr_db": sinr_values_db,
                        }
                    )
                )

            if eirp_dbm == reference_eirp_dbm and interferer_count in {0, 5, 20, 50}:
                power_rows.append(
                    {
                        "scenario": scenario.name,
                        "eirp_dbm": eirp_dbm,
                        "interferer_count": interferer_count,
                        "mean_signal_dbm": float(np.mean(signal_dbm)),
                        "mean_interference_dbm": float(mw_to_dbm(np.mean(dbm_to_mw(interference_dbm)))),
                        "noise_dbm": float(noise_dbm),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    powers = pd.DataFrame(power_rows)
    return summary, samples, powers


def run(config_path: Path, mitigation_path: Path | None, output_dir: Path) -> None:
    cfg = load_simulation_config(config_path)
    mitigation = load_mitigation_config(mitigation_path) if mitigation_path else None
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []
    powers: list[pd.DataFrame] = []

    for scenario in _scenario_list(mitigation):
        rng = np.random.default_rng(cfg.random_seed)
        summary, sinr_samples, power_breakdown = simulate_scenario(cfg, scenario, rng)
        summaries.append(summary)
        samples.append(sinr_samples)
        powers.append(power_breakdown)

    summary_output = pd.concat(summaries, ignore_index=True)
    if mitigation is None:
        summary_output.to_csv(output_dir / "baseline_results.csv", index=False)
    else:
        summary_output.to_csv(output_dir / "mitigation_results.csv", index=False)

    sample_output = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    power_output = pd.concat(powers, ignore_index=True) if powers else pd.DataFrame()

    sample_path = output_dir / "sinr_samples.csv"
    power_path = output_dir / "power_breakdown.csv"
    if sample_path.exists() and not sample_output.empty:
        existing = pd.read_csv(sample_path)
        sample_output = pd.concat([existing, sample_output], ignore_index=True)
    if power_path.exists() and not power_output.empty:
        existing = pd.read_csv(power_path)
        power_output = pd.concat([existing, power_output], ignore_index=True)
    sample_output.drop_duplicates(
        subset=["scenario", "eirp_dbm", "interferer_count", "sinr_db"],
        keep="last",
    ).to_csv(sample_path, index=False)
    power_output.drop_duplicates(
        subset=["scenario", "eirp_dbm", "interferer_count"],
        keep="last",
    ).to_csv(power_path, index=False)

    print(f"Wrote results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to baseline YAML config")
    parser.add_argument("--mitigation", type=Path, help="Optional mitigation YAML config")
    parser.add_argument("--output", type=Path, default=Path("results/data"), help="Output data directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, args.mitigation, args.output)


if __name__ == "__main__":
    main()
