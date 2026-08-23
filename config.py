"""Centralised, typed configuration loading.

Replaces the duplicated ``load_config`` + attribute-extraction blocks that previously
lived in ``strategy.py``. Everything is loaded once into frozen dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def _load_yaml(filepath: str | Path) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float
    max_power_kw: float
    efficiency_one_way: float
    soc_init: float
    soc_min: float
    soc_max: float
    degradation_cost_per_mwh: float
    settlement_period_hours: float
    periods_per_year: int

    @property
    def max_energy_per_period_kwh(self) -> float:
        """Energy movable in a single settlement period, given the power limit."""
        return self.max_power_kw * self.settlement_period_hours


@dataclass(frozen=True)
class TradingConfig:
    starting_balance: float
    transaction_cost_per_mwh: float
    min_edge_per_mwh: float


def load_battery_config(path: str | Path = ROOT / "BatteryConfig.yaml") -> BatteryConfig:
    raw = _load_yaml(path)
    b, m = raw["battery"], raw["market"]
    return BatteryConfig(
        capacity_kwh=b["capacity_kwh"],
        max_power_kw=b["max_power_kw"],
        efficiency_one_way=b["efficiency_one_way"],
        soc_init=b["soc_init"],
        soc_min=b["soc_min"],
        soc_max=b["soc_max"],
        degradation_cost_per_mwh=b["degradation_cost_per_mwh"],
        settlement_period_hours=m["settlement_period_hours"],
        periods_per_year=m["periods_per_year"],
    )


def load_trading_config(path: str | Path = ROOT / "TradingConfig.yaml") -> TradingConfig:
    t = _load_yaml(path)["trading"]
    return TradingConfig(
        starting_balance=t["starting_balance"],
        transaction_cost_per_mwh=t["transaction_cost_per_mwh"],
        min_edge_per_mwh=t["min_edge_per_mwh"],
    )
