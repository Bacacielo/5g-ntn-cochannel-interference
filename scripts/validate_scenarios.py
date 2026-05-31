"""Run scenario sanity checks for the 5G-NTN interference simulator.

These checks are not meant to prove a real constellation. They validate that the
model responds in physically expected directions across different operating
conditions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ntn_interference.config import load_simulation_config
from ntn_interference.simulation import Scenario, simulate_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
OUTPUT_MD = PROJECT_ROOT / "docs" / "scenario_validation.md"
OUTPUT_CSV = PROJECT_ROOT / "results" / "data" / "scenario_validation.csv"


def run_case(
    label: str,
    eirp_dbm: float,
    interferer_counts: list[int],
    bandwidth_hz: float = 5.0e6,
    altitude_m: float = 600_000,
    activity_factor: float = 1.0,
    beam_discrimination_db: float = 0.0,
    trials: int = 3000,
    seed: int = 2026,
) -> pd.DataFrame:
    base = load_simulation_config(BASELINE_CONFIG)
    cfg = replace(
        base,
        bandwidth_hz=bandwidth_hz,
        altitude_m=altitude_m,
        eirp_scenarios_dbm=[eirp_dbm],
        interferer_counts=interferer_counts,
        monte_carlo_trials=trials,
        outage_thresholds_db=[0],
        random_seed=seed,
    )
    scenario = Scenario(
        label,
        activity_factor=activity_factor,
        beam_discrimination_db=beam_discrimination_db,
    )
    summary, _, _ = simulate_scenario(cfg, scenario, np.random.default_rng(seed))
    return summary


def validation_rows() -> pd.DataFrame:
    frames = []

    frames.append(run_case("interferer_sweep", 60, [0, 5, 20, 50]))

    for suppression in [0, 5, 10, 15]:
        frames.append(
            run_case(
                f"beam_{suppression}_db",
                60,
                [20],
                beam_discrimination_db=suppression,
            )
        )

    for activity in [1.0, 0.5, 0.25]:
        frames.append(
            run_case(
                f"activity_{activity:.2f}",
                60,
                [20],
                activity_factor=activity,
            )
        )

    for bandwidth_mhz in [5, 10, 20]:
        frames.append(
            run_case(
                f"noise_limited_bw_{bandwidth_mhz}_mhz",
                60,
                [0],
                bandwidth_hz=bandwidth_mhz * 1e6,
            )
        )
        frames.append(
            run_case(
                f"interference_limited_bw_{bandwidth_mhz}_mhz",
                60,
                [20],
                bandwidth_hz=bandwidth_mhz * 1e6,
            )
        )

    return pd.concat(frames, ignore_index=True)


def assert_expected_behavior(results: pd.DataFrame) -> list[str]:
    notes = []

    sweep = results[results["scenario"] == "interferer_sweep"].sort_values("interferer_count")
    mean_sinr = sweep["mean_sinr_db"].to_numpy()
    outage = sweep["outage_lt_0_db"].to_numpy()
    assert np.all(np.diff(mean_sinr) < 0), "SINR should decrease as interferer count increases"
    assert np.all(np.diff(outage) >= 0), "outage should not decrease as interferer count increases"
    notes.append("PASS: increasing co-channel interferers lowers mean SINR and raises/non-decreases outage.")

    beam = results[results["scenario"].str.startswith("beam_")].copy()
    beam["suppression_db"] = beam["scenario"].str.extract(r"beam_(\d+)_db").astype(float)
    beam = beam.sort_values("suppression_db")
    assert np.all(np.diff(beam["mean_sinr_db"]) > 0), "beam discrimination should improve SINR"
    assert np.all(np.diff(beam["outage_lt_0_db"]) <= 0), "beam discrimination should not increase outage"
    notes.append("PASS: stronger beam discrimination improves SINR and lowers/non-increases outage.")

    activity = results[results["scenario"].str.startswith("activity_")].copy()
    activity["activity_factor"] = activity["scenario"].str.extract(r"activity_(\d+\.\d+)").astype(float)
    activity = activity.sort_values("activity_factor", ascending=False)
    assert np.all(np.diff(activity["mean_sinr_db"]) > 0), "lower activity should improve SINR"
    assert np.all(np.diff(activity["outage_lt_0_db"]) <= 0), "lower activity should not increase outage"
    notes.append("PASS: reducing co-channel activity improves SINR and lowers/non-increases outage.")

    noise_limited = results[results["scenario"].str.startswith("noise_limited_bw")].copy()
    noise_limited["bandwidth_mhz"] = (
        noise_limited["scenario"].str.extract(r"bw_(\d+)_mhz").astype(float)
    )
    noise_limited = noise_limited.sort_values("bandwidth_mhz")
    assert np.all(np.diff(noise_limited["mean_sinr_db"]) < 0), (
        "wider bandwidth should reduce noise-limited SINR"
    )
    notes.append("PASS: wider bandwidth reduces SINR in the no-interferer noise-limited case.")

    interference_limited = results[results["scenario"].str.startswith("interference_limited_bw")].copy()
    spread = interference_limited["mean_sinr_db"].max() - interference_limited["mean_sinr_db"].min()
    assert spread < 1.0, "bandwidth should have limited impact in the interference-limited case"
    notes.append("PASS: bandwidth has limited impact once aggregate interference dominates.")

    return notes


def write_markdown(results: pd.DataFrame, notes: list[str]) -> None:
    selected = results[
        [
            "scenario",
            "interferer_count",
            "mean_signal_dbm",
            "mean_interference_dbm",
            "noise_dbm",
            "mean_sinr_db",
            "outage_lt_0_db",
        ]
    ].copy()
    selected["mean_signal_dbm"] = selected["mean_signal_dbm"].round(2)
    selected["mean_interference_dbm"] = selected["mean_interference_dbm"].round(2)
    selected["noise_dbm"] = selected["noise_dbm"].round(2)
    selected["mean_sinr_db"] = selected["mean_sinr_db"].round(2)
    selected["outage_lt_0_db"] = selected["outage_lt_0_db"].round(4)

    table = _markdown_table(selected)

    text = [
        "# Scenario Validation",
        "",
        "This document records internal sanity checks for the simulator. These are",
        "not additional claims about a deployed constellation; they verify that the",
        "model responds in physically expected directions.",
        "",
        "## Checks",
        "",
    ]
    text.extend(f"- {note}" for note in notes)
    text.extend(
        [
            "",
            "## Scenario Results",
            "",
            table,
            "",
            "## Interpretation",
            "",
            "The validation cases support the current model implementation. More",
            "interferers degrade SINR, mitigation improves SINR, wider bandwidth",
            "hurts the noise-limited case, and bandwidth matters much less once",
            "aggregate interference dominates. These are the directions expected",
            "from the SINR equation and link-budget physics.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(text), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    results = validation_rows()
    notes = assert_expected_behavior(results)
    results.to_csv(OUTPUT_CSV, index=False)
    write_markdown(results, notes)
    print("docs/scenario_validation.md")
    print("results/data/scenario_validation.csv")


if __name__ == "__main__":
    main()
