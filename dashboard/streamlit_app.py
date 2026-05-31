"""Interactive Streamlit dashboard for the 5G-NTN interference simulator."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ntn_interference.config import SimulationConfig, load_simulation_config
from ntn_interference.simulation import Scenario, simulate_scenario


BASELINE_CONFIG = "configs/baseline.yaml"


@st.cache_data(show_spinner=False)
def run_dashboard_simulation(
    carrier_frequency_hz: float,
    bandwidth_hz: float,
    altitude_m: float,
    serving_elevation_deg: float,
    interferer_elevation_min_deg: float,
    interferer_elevation_max_deg: float,
    receiver_gain_dbi: float,
    receiver_noise_figure_db: float,
    misc_losses_db: float,
    serving_shadow_std_db: float,
    interferer_shadow_std_db: float,
    eirp_dbm: float,
    max_interferers: int,
    selected_interferers: int,
    trials: int,
    outage_threshold_db: float,
    activity_factor: float,
    beam_discrimination_db: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = load_simulation_config(BASELINE_CONFIG)
    counts = sorted(set(range(0, max_interferers + 1, 2)) | {selected_interferers})
    cfg = replace(
        base,
        carrier_frequency_hz=carrier_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        altitude_m=altitude_m,
        serving_elevation_deg=serving_elevation_deg,
        interferer_elevation_min_deg=interferer_elevation_min_deg,
        interferer_elevation_max_deg=interferer_elevation_max_deg,
        receiver_gain_dbi=receiver_gain_dbi,
        receiver_noise_figure_db=receiver_noise_figure_db,
        misc_losses_db=misc_losses_db,
        serving_shadow_std_db=serving_shadow_std_db,
        interferer_shadow_std_db=interferer_shadow_std_db,
        eirp_scenarios_dbm=[eirp_dbm],
        interferer_counts=counts,
        monte_carlo_trials=trials,
        outage_thresholds_db=[outage_threshold_db],
        random_seed=random_seed,
    )
    scenario = Scenario(
        "interactive",
        activity_factor=activity_factor,
        beam_discrimination_db=beam_discrimination_db,
    )
    return simulate_scenario(
        cfg,
        scenario,
        np.random.default_rng(random_seed),
        sample_interferer_counts={selected_interferers},
    )


@st.cache_data(show_spinner=False)
def run_comparison_simulation(
    carrier_frequency_hz: float,
    bandwidth_hz: float,
    altitude_m: float,
    serving_elevation_deg: float,
    receiver_noise_figure_db: float,
    eirp_dbm: float,
    max_interferers: int,
    selected_interferers: int,
    trials: int,
    outage_threshold_db: float,
    random_seed: int,
) -> pd.DataFrame:
    base = load_simulation_config(BASELINE_CONFIG)
    counts = sorted(set(range(0, max_interferers + 1, 2)) | {selected_interferers})
    cfg = replace(
        base,
        carrier_frequency_hz=carrier_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        altitude_m=altitude_m,
        serving_elevation_deg=serving_elevation_deg,
        receiver_noise_figure_db=receiver_noise_figure_db,
        eirp_scenarios_dbm=[eirp_dbm],
        interferer_counts=counts,
        monte_carlo_trials=trials,
        outage_thresholds_db=[outage_threshold_db],
        random_seed=random_seed,
    )
    scenarios = [
        Scenario("baseline"),
        Scenario("activity_50_percent", activity_factor=0.5),
        Scenario("beam_10_db", beam_discrimination_db=10.0),
        Scenario("beam_15_db", beam_discrimination_db=15.0),
    ]
    frames = []
    for scenario in scenarios:
        summary, _, _ = simulate_scenario(
            cfg,
            scenario,
            np.random.default_rng(random_seed),
            sample_interferer_counts=set(),
        )
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def outage_column(threshold_db: float) -> str:
    return f"outage_lt_{threshold_db:g}_db"


def plot_outage(summary: pd.DataFrame, threshold_db: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")
    summary = summary.sort_values("interferer_count")
    ax.plot(
        summary["interferer_count"],
        summary[outage_column(threshold_db)],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Co-channel interferers")
    ax.set_ylabel(f"Outage probability (SINR < {threshold_db:g} dB)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def plot_sinr_distribution(samples: pd.DataFrame, selected_interferers: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")
    values = samples["sinr_db"]
    ax.hist(values, bins=40, density=True, color="#1f77b4", alpha=0.78)
    ax.axvline(values.mean(), color="#d62728", linewidth=2, label=f"Mean {values.mean():.2f} dB")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"SINR Distribution at {selected_interferers} Interferers")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_power_breakdown(case: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")
    labels = ["Signal", "Interference", "Noise"]
    values = [
        case["mean_signal_dbm"],
        case["mean_interference_dbm"],
        case["noise_dbm"],
    ]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Power (dBm)")
    ax.set_title("Mean Received Power Terms")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    return fig


def plot_comparison(comparison: pd.DataFrame, threshold_db: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")
    scenario_order = ["baseline", "activity_50_percent", "beam_10_db", "beam_15_db"]
    for scenario in scenario_order:
        group = comparison[comparison["scenario"] == scenario]
        if group.empty:
            continue
        group = group.sort_values("interferer_count")
        ax.plot(
            group["interferer_count"],
            group[outage_column(threshold_db)],
            marker="o",
            linewidth=2,
            label=scenario.replace("_", " "),
        )
    ax.set_xlabel("Co-channel interferers")
    ax.set_ylabel(f"Outage probability (SINR < {threshold_db:g} dB)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="5G-NTN Interference",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: rgba(128, 128, 128, 0.055);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("5G-NTN Co-Channel Interference")
    st.caption(
        "First-order Monte Carlo link-budget model for SINR, outage, and "
        "interference-mitigation sensitivity."
    )

    with st.sidebar:
        st.header("Scenario Controls")
        with st.expander("Link budget", expanded=True):
            eirp_dbm = st.slider("Satellite EIRP (dBm)", 45.0, 65.0, 60.0, 1.0)
            altitude_km = st.slider("LEO altitude (km)", 500, 1400, 600, 50)
            bandwidth_mhz = st.select_slider("Bandwidth (MHz)", [5, 10, 15, 20], value=5)
            carrier_ghz = st.slider("Carrier frequency (GHz)", 1.5, 2.5, 2.0, 0.1)
            serving_elevation_deg = st.slider("Serving elevation (deg)", 20, 90, 60, 5)

        with st.expander("Interference", expanded=True):
            max_interferers = st.slider("Maximum interferers in sweep", 10, 80, 50, 2)
            selected_interferers = st.slider("Selected interferer count", 0, max_interferers, 20, 1)
            activity_factor = st.slider("Co-channel activity factor", 0.0, 1.0, 1.0, 0.05)
            beam_discrimination_db = st.slider("Beam discrimination (dB)", 0.0, 25.0, 0.0, 1.0)

        with st.expander("Receiver and Monte Carlo", expanded=False):
            receiver_noise_figure_db = st.slider("Receiver noise figure (dB)", 2.0, 12.0, 7.0, 0.5)
            trials = st.select_slider("Monte Carlo trials", [500, 1000, 2000, 5000, 10000], value=2000)
            outage_threshold_db = st.select_slider("Outage threshold (dB)", [-3.0, 0.0, 3.0], value=0.0)
            random_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    with st.spinner("Running Monte Carlo simulation..."):
        summary, samples, powers = run_dashboard_simulation(
            carrier_frequency_hz=carrier_ghz * 1e9,
            bandwidth_hz=bandwidth_mhz * 1e6,
            altitude_m=altitude_km * 1000.0,
            serving_elevation_deg=float(serving_elevation_deg),
            interferer_elevation_min_deg=10.0,
            interferer_elevation_max_deg=60.0,
            receiver_gain_dbi=0.0,
            receiver_noise_figure_db=float(receiver_noise_figure_db),
            misc_losses_db=2.0,
            serving_shadow_std_db=2.0,
            interferer_shadow_std_db=4.0,
            eirp_dbm=float(eirp_dbm),
            max_interferers=int(max_interferers),
            selected_interferers=int(selected_interferers),
            trials=int(trials),
            outage_threshold_db=float(outage_threshold_db),
            activity_factor=float(activity_factor),
            beam_discrimination_db=float(beam_discrimination_db),
            random_seed=int(random_seed),
        )

    selected_case = summary[summary["interferer_count"] == selected_interferers].iloc[0]

    st.write(
        "This dashboard is an exploratory prototype. The submission PDF should use "
        "the clean static figures generated in `report/figures/`."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean SINR", f"{selected_case['mean_sinr_db']:.2f} dB")
    col2.metric("Median SINR", f"{selected_case['median_sinr_db']:.2f} dB")
    col3.metric("Outage probability", f"{selected_case[outage_column(outage_threshold_db)]:.3f}")
    col4.metric("Noise power", f"{selected_case['noise_dbm']:.2f} dBm")

    comparison = run_comparison_simulation(
        carrier_frequency_hz=carrier_ghz * 1e9,
        bandwidth_hz=bandwidth_mhz * 1e6,
        altitude_m=altitude_km * 1000.0,
        serving_elevation_deg=float(serving_elevation_deg),
        receiver_noise_figure_db=float(receiver_noise_figure_db),
        eirp_dbm=float(eirp_dbm),
        max_interferers=int(max_interferers),
        selected_interferers=int(selected_interferers),
        trials=int(trials),
        outage_threshold_db=float(outage_threshold_db),
        random_seed=int(random_seed),
    )

    overview_tab, mitigation_tab, data_tab = st.tabs(["Overview", "Mitigation", "Data"])

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_outage(summary, outage_threshold_db), clear_figure=True)
        with right:
            st.pyplot(plot_sinr_distribution(samples, selected_interferers), clear_figure=True)
        st.pyplot(plot_power_breakdown(selected_case), clear_figure=True)

    with mitigation_tab:
        st.pyplot(plot_comparison(comparison, outage_threshold_db), clear_figure=True)
        comparison_case = comparison[comparison["interferer_count"] == selected_interferers].copy()
        comparison_case = comparison_case[
            [
                "scenario",
                "mean_sinr_db",
                "median_sinr_db",
                outage_column(outage_threshold_db),
                "mean_signal_dbm",
                "mean_interference_dbm",
            ]
        ]
        comparison_case = comparison_case.rename(
            columns={
                "scenario": "Scenario",
                "mean_sinr_db": "Mean SINR (dB)",
                "median_sinr_db": "Median SINR (dB)",
                outage_column(outage_threshold_db): "Outage probability",
                "mean_signal_dbm": "Mean signal (dBm)",
                "mean_interference_dbm": "Mean interference (dBm)",
            }
        )
        comparison_case["Scenario"] = comparison_case["Scenario"].str.replace("_", " ", regex=False)
        st.dataframe(comparison_case, use_container_width=True, hide_index=True)

    with data_tab:
        st.subheader("Selected Case")
        st.dataframe(
            selected_case[
                [
                    "eirp_dbm",
                    "interferer_count",
                    "mean_signal_dbm",
                    "mean_interference_dbm",
                    "noise_dbm",
                    "mean_sinr_db",
                    "median_sinr_db",
                    outage_column(outage_threshold_db),
                ]
            ].to_frame("value"),
            use_container_width=True,
        )

        st.subheader("Sweep Results")
        st.dataframe(
            summary[
                [
                    "interferer_count",
                    "mean_sinr_db",
                    "median_sinr_db",
                    "p05_sinr_db",
                    "p95_sinr_db",
                    outage_column(outage_threshold_db),
                ]
            ].sort_values("interferer_count"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
