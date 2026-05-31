"""Generate clean report-ready figures from simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures"
REPORT_EIRP_DBM = 60.0
OUTAGE_COLUMN = "outage_lt_0_db"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f2f2f",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#8a8a8a",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(DATA_DIR / "baseline_results.csv")
    mitigation = pd.read_csv(DATA_DIR / "mitigation_results.csv")
    baseline = baseline[(baseline["scenario"] == "baseline") & (baseline["eirp_dbm"] == REPORT_EIRP_DBM)]
    mitigation = mitigation[mitigation["eirp_dbm"] == REPORT_EIRP_DBM]
    return baseline.sort_values("interferer_count"), mitigation.sort_values("interferer_count")


def system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    ue = Circle((5, 1.25), 0.35, facecolor="#f5f5f5", edgecolor="#222222", linewidth=1.4)
    ax.add_patch(ue)
    ax.text(5, 0.55, "Δέκτης UE-θύμα", ha="center", va="center", fontsize=10)

    nodes = [
        ((2.0, 4.75), "Εξυπηρετών\nδορυφόρος", "#1f77b4", "Επιθυμητός φορέας C"),
        ((5.0, 5.0), "Συγκαναλικός\nπαρεμβολέας", "#d62728", "Παρεμβολή I1"),
        ((8.0, 4.35), "Συγκαναλικός\nπαρεμβολέας", "#d62728", "Παρεμβολή I2"),
    ]

    for (x, y), label, color, arrow_label in nodes:
        sat = Circle((x, y), 0.32, facecolor=color, edgecolor="#222222", linewidth=1.0, alpha=0.92)
        ax.add_patch(sat)
        ax.text(x, y + 0.68, label, ha="center", va="center", fontsize=9)
        arrow = FancyArrowPatch(
            (x, y - 0.35),
            (5, 1.62),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            alpha=0.9,
        )
        ax.add_patch(arrow)
        mid_x = (x + 5) / 2
        mid_y = (y + 1.7) / 2
        ax.text(mid_x, mid_y, arrow_label, color=color, fontsize=8, ha="center", va="center")

    ax.text(6.25, 1.2, "Θερμικός θόρυβος N", color="#555555", fontsize=9, va="center")
    ax.annotate("", xy=(5.37, 1.25), xytext=(6.1, 1.25), arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_title("Σενάριο Συγκαναλικής Παρεμβολής Κατερχόμενης Ζεύξης", pad=14)
    _save(fig, "system_interference_diagram.png")


def sinr_trend(baseline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        baseline["interferer_count"],
        baseline["median_sinr_db"],
        color="#1f77b4",
        linewidth=2.2,
        label="Διάμεσο SINR",
    )
    ax.fill_between(
        baseline["interferer_count"],
        baseline["p05_sinr_db"],
        baseline["p95_sinr_db"],
        color="#1f77b4",
        alpha=0.16,
        label="5ο-95ο εκατοστημόριο",
    )
    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", label="Κατώφλι 0 dB")
    ax.set_title(f"Υποβάθμιση SINR στο βασικό σενάριο ({REPORT_EIRP_DBM:g} dBm EIRP)")
    ax.set_xlabel("Ενεργοί συγκαναλικοί παρεμβολείς")
    ax.set_ylabel("SINR (dB)")
    ax.legend(loc="upper right")
    _save(fig, "baseline_sinr_vs_interferers.png")


def outage_trend(baseline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        baseline["interferer_count"],
        baseline[OUTAGE_COLUMN],
        color="#d62728",
        linewidth=2.2,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Αύξηση πιθανότητας διακοπής στο βασικό σενάριο ({REPORT_EIRP_DBM:g} dBm EIRP)")
    ax.set_xlabel("Ενεργοί συγκαναλικοί παρεμβολείς")
    ax.set_ylabel("Πιθανότητα διακοπής, SINR < 0 dB")
    _save(fig, "baseline_outage_vs_interferers.png")


def power_terms(baseline: pd.DataFrame) -> None:
    selected = baseline[baseline["interferer_count"].isin([0, 5, 20, 50])].copy()
    selected.loc[selected["interferer_count"] == 0, "mean_interference_dbm"] = float("nan")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(selected["interferer_count"], selected["mean_signal_dbm"], marker="o", label="Επιθυμητό σήμα C")
    ax.plot(
        selected["interferer_count"],
        selected["mean_interference_dbm"],
        marker="o",
        label="Συνολική παρεμβολή I",
    )
    ax.plot(selected["interferer_count"], selected["noise_dbm"], marker="o", label="Θερμικός θόρυβος N")
    ax.set_title(f"Όροι λαμβανόμενης ισχύος ({REPORT_EIRP_DBM:g} dBm EIRP)")
    ax.set_xlabel("Ενεργοί συγκαναλικοί παρεμβολείς")
    ax.set_ylabel("Λαμβανόμενη ισχύς (dBm)")
    ax.set_ylim(-105, -80)
    ax.legend(loc="best")
    _save(fig, "received_power_terms.png")


def mitigation_comparison(mitigation: pd.DataFrame) -> None:
    labels = {
        "activity_reduction": "50% δραστηριότητα",
        "beam_discrimination": "10 dB διάκριση δέσμης",
    }
    baseline, _ = _load_summary()
    baseline_case = baseline.copy()
    baseline_case["scenario"] = "baseline"
    combined = pd.concat([baseline_case, mitigation], ignore_index=True)
    colors = {
        "baseline": "#d62728",
        "activity_reduction": "#ff7f0e",
        "beam_discrimination": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for scenario in ["baseline", "activity_reduction", "beam_discrimination"]:
        group = combined[combined["scenario"] == scenario].sort_values("interferer_count")
        ax.plot(
            group["interferer_count"],
            group[OUTAGE_COLUMN],
            linewidth=2.2,
            label=labels.get(scenario, "Βασικό"),
            color=colors[scenario],
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Ευαισθησία σε μηχανισμούς μείωσης ({REPORT_EIRP_DBM:g} dBm EIRP)")
    ax.set_xlabel("Ενεργοί συγκαναλικοί παρεμβολείς")
    ax.set_ylabel("Πιθανότητα διακοπής, SINR < 0 dB")
    ax.legend(loc="lower right")
    _save(fig, "mitigation_sensitivity_outage.png")


def sinr_cdf() -> None:
    samples = pd.read_csv(DATA_DIR / "sinr_samples.csv")
    samples = samples[(samples["scenario"] == "baseline") & (samples["eirp_dbm"] == REPORT_EIRP_DBM)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for count, color in [(5, "#1f77b4"), (20, "#ff7f0e"), (50, "#2ca02c")]:
        group = samples[samples["interferer_count"] == count]
        values = group["sinr_db"].sort_values().reset_index(drop=True)
        probability = (values.index + 1) / len(values)
        ax.plot(values, probability, linewidth=2.2, label=f"{count} παρεμβολείς", color=color)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", label="Κατώφλι 0 dB")
    ax.set_title(f"Κατανομή SINR στο βασικό σενάριο ({REPORT_EIRP_DBM:g} dBm EIRP)")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right")
    _save(fig, "baseline_sinr_cdf.png")


def main() -> None:
    _style()
    baseline, mitigation = _load_summary()
    system_diagram()
    sinr_trend(baseline)
    outage_trend(baseline)
    power_terms(baseline)
    mitigation_comparison(mitigation)
    sinr_cdf()
    print("Wrote report figures to report/figures")


if __name__ == "__main__":
    main()
