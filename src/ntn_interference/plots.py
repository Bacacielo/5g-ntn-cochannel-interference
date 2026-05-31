"""Plotting CLI for 5G-NTN interference simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("baseline_results.csv", "mitigation_results.csv"):
        path = results_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _outage_column(df: pd.DataFrame) -> str:
    for candidate in ("outage_lt_0_db", "outage_lt_3_db"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No outage column found")


def plot_outage_vs_interferers(df: pd.DataFrame, output_dir: Path) -> None:
    outage_col = _outage_column(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (scenario, eirp), group in df.groupby(["scenario", "eirp_dbm"]):
        group = group.sort_values("interferer_count")
        ax.plot(group["interferer_count"], group[outage_col], label=f"{scenario}, {eirp:g} dBm")
    ax.set_xlabel("Number of co-channel interferers")
    ax.set_ylabel(f"Outage probability ({outage_col.replace('_', ' ')})")
    ax.set_title("Outage Probability vs Co-Channel Interferers")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "outage_vs_interferers.png", dpi=200)
    plt.close(fig)


def plot_average_sinr_vs_interferers(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (scenario, eirp), group in df.groupby(["scenario", "eirp_dbm"]):
        group = group.sort_values("interferer_count")
        ax.plot(group["interferer_count"], group["mean_sinr_db"], label=f"{scenario}, {eirp:g} dBm")
    ax.set_xlabel("Number of co-channel interferers")
    ax.set_ylabel("Average SINR (dB)")
    ax.set_title("Average SINR vs Co-Channel Interferers")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "average_sinr_vs_interferers.png", dpi=200)
    plt.close(fig)


def plot_sinr_cdf(results_dir: Path, output_dir: Path) -> None:
    sample_path = results_dir / "sinr_samples.csv"
    if not sample_path.exists():
        return
    samples = pd.read_csv(sample_path)
    if samples.empty:
        return
    middle_eirp = sorted(samples["eirp_dbm"].unique())[len(samples["eirp_dbm"].unique()) // 2]
    samples = samples[(samples["eirp_dbm"] == middle_eirp) & (samples["scenario"] == "baseline")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for count, group in samples.groupby("interferer_count"):
        values = group["sinr_db"].sort_values().reset_index(drop=True)
        probability = (values.index + 1) / len(values)
        ax.plot(values, probability, label=f"{int(count)} interferers")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Baseline SINR CDF at {middle_eirp:g} dBm EIRP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sinr_cdf.png", dpi=200)
    plt.close(fig)


def plot_power_breakdown(results_dir: Path, output_dir: Path) -> None:
    power_path = results_dir / "power_breakdown.csv"
    if not power_path.exists():
        return
    power = pd.read_csv(power_path)
    power = power[power["scenario"] == "baseline"].sort_values("interferer_count")
    if power.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(power["interferer_count"], power["mean_signal_dbm"], marker="o", label="Signal")
    ax.plot(power["interferer_count"], power["mean_interference_dbm"], marker="o", label="Interference")
    ax.plot(power["interferer_count"], power["noise_dbm"], marker="o", label="Noise")
    ax.set_xlabel("Number of co-channel interferers")
    ax.set_ylabel("Power (dBm)")
    ax.set_title("Received Power Breakdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "power_breakdown.png", dpi=200)
    plt.close(fig)


def plot_mitigation_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    outage_col = _outage_column(df)
    eirp_values = sorted(df["eirp_dbm"].unique())
    middle_eirp = eirp_values[len(eirp_values) // 2]
    subset = df[df["eirp_dbm"] == middle_eirp]
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, group in subset.groupby("scenario"):
        group = group.sort_values("interferer_count")
        ax.plot(group["interferer_count"], group[outage_col], label=scenario)
    ax.set_xlabel("Number of co-channel interferers")
    ax.set_ylabel(f"Outage probability ({outage_col.replace('_', ' ')})")
    ax.set_title(f"Mitigation Comparison at {middle_eirp:g} dBm EIRP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mitigation_comparison.png", dpi=200)
    plt.close(fig)


def run(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_results(results_dir)
    plot_outage_vs_interferers(df, output_dir)
    plot_average_sinr_vs_interferers(df, output_dir)
    plot_sinr_cdf(results_dir, output_dir)
    plot_power_breakdown(results_dir, output_dir)
    plot_mitigation_comparison(df, output_dir)
    print(f"Wrote figures to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Directory containing result CSV files")
    parser.add_argument("--output", type=Path, default=Path("results/figures"), help="Output figure directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.results, args.output)


if __name__ == "__main__":
    main()

