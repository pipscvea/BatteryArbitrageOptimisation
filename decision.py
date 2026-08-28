"""Stage 3 — convert a forecast into a trading decision.

The forecast alone is not a strategy. Here we turn a predicted forward price change into
an *expected round-trip edge* (£/MWh, net of efficiency loss and costs) and only trade
when that edge clears ``min_edge_per_mwh``. Physical feasibility (SoC / power limits) is
enforced downstream in ``simulate.py``; this layer decides intent and desired volume.

Returned request (kWh, per settlement period):
    > 0  => charge  (buy from grid now, expecting to discharge higher later)
    < 0  => discharge (sell to grid now, expecting to recharge cheaper later)
    0    => hold
"""
from __future__ import annotations

from dataclasses import dataclass

from config import BatteryConfig, TradingConfig


@dataclass
class Decision:
    request_kwh: float   # signed: +charge / -discharge / 0 hold
    expected_edge: float  # £/MWh net edge that justified the trade (0 if hold)


def _mwh(kwh: float) -> float:
    return kwh / 1000.0


def _sized_volume(edge: float, batt: BatteryConfig, trade: TradingConfig,
                  size_scale: float) -> float:
    """Confidence-proportional volume: ramp from 0 at the min-edge gate up to full power
    once the edge exceeds the gate by ``size_scale`` £/MWh. ``size_scale <= 0`` restores
    all-or-nothing sizing (full power the moment the gate is cleared)."""
    full = batt.max_energy_per_period_kwh
    if size_scale <= 0:
        return full
    frac = (edge - trade.min_edge_per_mwh) / size_scale
    return full * min(max(frac, 0.0), 1.0)


def decide(forecast_change: float, price_sell: float, price_buy: float,
           batt: BatteryConfig, trade: TradingConfig, size_scale: float = 0.0) -> Decision:
    """Decision from a single forecast of the forward price change (£/MWh).

    charge edge  = eff * E[future_sell] - buy_now - costs
    discharge edge = sell_now - E[future_buy] / eff - costs
    where E[future] = current + forecast_change. Volume scales with the edge above the
    gate when ``size_scale > 0`` (trade bigger on stronger signals), else full power.
    """
    eff = batt.efficiency_one_way
    costs = trade.transaction_cost_per_mwh + batt.degradation_cost_per_mwh
    exp_future_sell = price_sell + forecast_change
    exp_future_buy = price_buy + forecast_change

    charge_edge = eff * exp_future_sell - price_buy - costs
    discharge_edge = price_sell - exp_future_buy / eff - costs

    if charge_edge >= trade.min_edge_per_mwh and charge_edge >= discharge_edge:
        return Decision(+_sized_volume(charge_edge, batt, trade, size_scale), charge_edge)
    if discharge_edge >= trade.min_edge_per_mwh:
        return Decision(-_sized_volume(discharge_edge, batt, trade, size_scale), discharge_edge)
    return Decision(request_kwh=0.0, expected_edge=0.0)
