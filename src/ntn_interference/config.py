"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    carrier_frequency_hz: float
    bandwidth_hz: float
    altitude_m: float
    serving_elevation_deg: float
    interferer_elevation_min_deg: float
    interferer_elevation_max_deg: float
    receiver_gain_dbi: float
    receiver_noise_figure_db: float
    misc_losses_db: float
    serving_shadow_std_db: float
    interferer_shadow_std_db: float
    eirp_scenarios_dbm: list[float]
    interferer_counts: list[int]
    monte_carlo_trials: int
    outage_thresholds_db: list[float]
    random_seed: int


@dataclass(frozen=True)
class MitigationConfig:
    activity_factor: float
    beam_discrimination_db: float


def _as_float(data: dict[str, Any], key: str) -> float:
    try:
        return float(data[key])
    except KeyError as exc:
        raise ValueError(f"missing required config key: {key}") from exc


def _as_int(data: dict[str, Any], key: str) -> int:
    try:
        return int(data[key])
    except KeyError as exc:
        raise ValueError(f"missing required config key: {key}") from exc


def _parse_interferer_counts(value: Any) -> list[int]:
    if isinstance(value, dict):
        start = int(value["start"])
        stop = int(value["stop"])
        step = int(value.get("step", 1))
        return list(range(start, stop + 1, step))
    if isinstance(value, list):
        return [int(v) for v in value]
    raise ValueError("interferer_counts must be a list or {start, stop, step} mapping")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_simulation_config(path: str | Path) -> SimulationConfig:
    data = load_yaml(path)
    cfg = SimulationConfig(
        carrier_frequency_hz=_as_float(data, "carrier_frequency_hz"),
        bandwidth_hz=_as_float(data, "bandwidth_hz"),
        altitude_m=_as_float(data, "altitude_m"),
        serving_elevation_deg=_as_float(data, "serving_elevation_deg"),
        interferer_elevation_min_deg=_as_float(data, "interferer_elevation_min_deg"),
        interferer_elevation_max_deg=_as_float(data, "interferer_elevation_max_deg"),
        receiver_gain_dbi=_as_float(data, "receiver_gain_dbi"),
        receiver_noise_figure_db=_as_float(data, "receiver_noise_figure_db"),
        misc_losses_db=_as_float(data, "misc_losses_db"),
        serving_shadow_std_db=_as_float(data, "serving_shadow_std_db"),
        interferer_shadow_std_db=_as_float(data, "interferer_shadow_std_db"),
        eirp_scenarios_dbm=[float(v) for v in data["eirp_scenarios_dbm"]],
        interferer_counts=_parse_interferer_counts(data["interferer_counts"]),
        monte_carlo_trials=_as_int(data, "monte_carlo_trials"),
        outage_thresholds_db=[float(v) for v in data["outage_thresholds_db"]],
        random_seed=_as_int(data, "random_seed"),
    )
    validate_simulation_config(cfg)
    return cfg


def load_mitigation_config(path: str | Path) -> MitigationConfig:
    data = load_yaml(path)
    cfg = MitigationConfig(
        activity_factor=_as_float(data, "activity_factor"),
        beam_discrimination_db=_as_float(data, "beam_discrimination_db"),
    )
    if not 0.0 <= cfg.activity_factor <= 1.0:
        raise ValueError("activity_factor must be between 0 and 1")
    if cfg.beam_discrimination_db < 0:
        raise ValueError("beam_discrimination_db must be non-negative")
    return cfg


def validate_simulation_config(cfg: SimulationConfig) -> None:
    positive_values = {
        "carrier_frequency_hz": cfg.carrier_frequency_hz,
        "bandwidth_hz": cfg.bandwidth_hz,
        "altitude_m": cfg.altitude_m,
        "monte_carlo_trials": cfg.monte_carlo_trials,
    }
    for key, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive")
    if not cfg.eirp_scenarios_dbm:
        raise ValueError("eirp_scenarios_dbm must not be empty")
    if not cfg.interferer_counts:
        raise ValueError("interferer_counts must not be empty")
    if any(count < 0 for count in cfg.interferer_counts):
        raise ValueError("interferer_counts must be non-negative")
    if cfg.interferer_elevation_min_deg >= cfg.interferer_elevation_max_deg:
        raise ValueError("interferer elevation min must be smaller than max")
    if cfg.receiver_noise_figure_db < 0:
        raise ValueError("receiver_noise_figure_db must be non-negative")

