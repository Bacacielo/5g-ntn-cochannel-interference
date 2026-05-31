"""Sensitivity-analysis CLI for the 5G-NTN interference model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ntn_interference.config import SimulationConfig, load_simulation_config, load_yaml
from ntn_interference.simulation import Scenario, simulate_scenario


def _load_sensitivity_config(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    required = {
        "eirp_dbm",
        "interferer_count",
        "outage_threshold_db",
        "altitude_sweep_m",
        "bandwidth_sweep_hz",
        "beam_discrimination_sweep_db",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"missing sensitivity config keys: {', '.join(missing)}")
    return data


def _single_case(
    cfg: SimulationConfig,
    scenario: Scenario,
    eirp_dbm: float,
    interferer_count: int,
    rng: np.random.Generator,
) -> pd.Series:
    case_cfg = replace(cfg, eirp_scenarios_dbm=[eirp_dbm], interferer_counts=[interferer_count])
    summary, _, _ = simulate_scenario(case_cfg, scenario, rng)
    return summary.iloc[0]


def _sweep_altitude(
    cfg: SimulationConfig,
    sensitivity: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for altitude_m in sensitivity["altitude_sweep_m"]:
        case = _single_case(
            replace(cfg, altitude_m=float(altitude_m)),
            Scenario("baseline"),
            float(sensitivity["eirp_dbm"]),
            int(sensitivity["interferer_count"]),
            rng,
        )
        rows.append(_result_row("altitude_km", float(altitude_m) / 1000.0, case, sensitivity))
    return rows


def _sweep_bandwidth(
    cfg: SimulationConfig,
    sensitivity: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for bandwidth_hz in sensitivity["bandwidth_sweep_hz"]:
        case = _single_case(
            replace(cfg, bandwidth_hz=float(bandwidth_hz)),
            Scenario("baseline"),
            float(sensitivity["eirp_dbm"]),
            int(sensitivity["interferer_count"]),
            rng,
        )
        rows.append(_result_row("bandwidth_mhz", float(bandwidth_hz) / 1.0e6, case, sensitivity))
    return rows


def _sweep_beam_discrimination(
    cfg: SimulationConfig,
    sensitivity: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for suppression_db in sensitivity["beam_discrimination_sweep_db"]:
        case = _single_case(
            cfg,
            Scenario("beam_discrimination", beam_discrimination_db=float(suppression_db)),
            float(sensitivity["eirp_dbm"]),
            int(sensitivity["interferer_count"]),
            rng,
        )
        rows.append(_result_row("beam_discrimination_db", float(suppression_db), case, sensitivity))
    return rows


def _result_row(
    sweep: str,
    value: float,
    case: pd.Series,
    sensitivity: dict[str, Any],
) -> dict[str, float | str]:
    threshold = float(sensitivity["outage_threshold_db"])
    outage_column = f"outage_lt_{threshold:g}_db"
    return {
        "sweep": sweep,
        "value": value,
        "eirp_dbm": float(case["eirp_dbm"]),
        "interferer_count": int(case["interferer_count"]),
        "mean_sinr_db": float(case["mean_sinr_db"]),
        "median_sinr_db": float(case["median_sinr_db"]),
        "outage_probability": float(case[outage_column]),
        "mean_signal_dbm": float(case["mean_signal_dbm"]),
        "mean_interference_dbm": float(case["mean_interference_dbm"]),
        "noise_dbm": float(case["noise_dbm"]),
    }


def run(
    baseline_config_path: Path,
    sensitivity_config_path: Path,
    output_data_dir: Path,
    output_figure_dir: Path,
) -> None:
    cfg = load_simulation_config(baseline_config_path)
    sensitivity = _load_sensitivity_config(sensitivity_config_path)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed + 1000)
    rows = []
    rows.extend(_sweep_altitude(cfg, sensitivity, rng))
    rows.extend(_sweep_bandwidth(cfg, sensitivity, rng))
    rows.extend(_sweep_beam_discrimination(cfg, sensitivity, rng))

    results = pd.DataFrame(rows)
    results.to_csv(output_data_dir / "sensitivity_results.csv", index=False)
    plot_sensitivity(results, output_figure_dir / "sensitivity_analysis.png")
    print("Wrote sensitivity results and figure")


def plot_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    labels = {
        "altitude_km": ("Altitude (km)", "Altitude Sensitivity"),
        "bandwidth_mhz": ("Bandwidth (MHz)", "Bandwidth Sensitivity"),
        "beam_discrimination_db": ("Beam discrimination (dB)", "Beam Discrimination Sensitivity"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, sweep in zip(axes, labels):
        subset = results[results["sweep"] == sweep].sort_values("value")
        xlabel, title = labels[sweep]
        ax.plot(subset["value"], subset["mean_sinr_db"], marker="o", color="#1f77b4")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean SINR (dB)", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        outage_axis = ax.twinx()
        outage_axis.plot(
            subset["value"],
            subset["outage_probability"],
            marker="s",
            color="#d62728",
        )
        outage_axis.set_ylabel("Outage probability", color="#d62728")
        outage_axis.tick_params(axis="y", labelcolor="#d62728")
        outage_axis.set_ylim(-0.05, 1.05)

    fig.suptitle("Sensitivity Analysis at 60 dBm EIRP and 20 Co-Channel Interferers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to baseline YAML config")
    parser.add_argument(
        "--sensitivity",
        required=True,
        type=Path,
        help="Path to sensitivity YAML config",
    )
    parser.add_argument("--output-data", type=Path, default=Path("results/data"))
    parser.add_argument("--output-figures", type=Path, default=Path("results/figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, args.sensitivity, args.output_data, args.output_figures)


if __name__ == "__main__":
    main()

