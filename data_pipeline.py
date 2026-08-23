"""Stage 0/2 — assemble the market dataset.

Loads UK imbalance price CSVs and merges them with system demand into a single
datetime-indexed frame. Cross-platform paths (was hardcoded Windows backslashes).

Expected on-disk layout (NOT committed to the repo — see README for how to obtain):

    SystemSellAndBuyPrices/SystemSellAndBuyPrices-2017*.csv
        columns: StartTime, SystemSellPrice, SystemBuyPrice   (£/MWh)
    RollingSystemDemand/RollingSystemDemand-*.csv
        columns: StartTime, Demand                            (MW)

Stage 2 driver data (wind/solar/gas/carbon/interconnectors/weather) would be merged
here too, via the same datetime index — see ``merge_driver``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PRICE_DIR = ROOT / "SystemSellAndBuyPrices"
DEMAND_DIR = ROOT / "RollingSystemDemand"


def merge_price_csvs(file_list, datetime_col: str = "StartTime") -> pd.DataFrame:
    """Concatenate price CSVs into one sorted, deduplicated, datetime-indexed frame.

    Zero System Sell Prices are treated as missing and linearly interpolated from
    their neighbours (a data-quality fix carried over from the original code).
    """
    if not file_list:
        raise FileNotFoundError(
            f"No price CSVs found. Expected files under {PRICE_DIR} (see README)."
        )
    frames = [pd.read_csv(f, parse_dates=[datetime_col]) for f in file_list]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=datetime_col).sort_values(by=datetime_col)
    merged = merged.set_index(datetime_col)
    zero = merged["SystemSellPrice"] == 0
    merged["SystemSellPrice"] = merged["SystemSellPrice"].mask(
        zero, (merged["SystemSellPrice"].shift() + merged["SystemSellPrice"].shift(-1)) / 2
    )
    return merged


def merge_with_demand(
    price_df: pd.DataFrame, demand_csv, datetime_col: str = "StartTime", how: str = "inner"
) -> pd.DataFrame:
    """Join a price frame with a demand CSV on the datetime index."""
    demand_df = pd.read_csv(demand_csv, parse_dates=[datetime_col]).set_index(datetime_col)
    merged = price_df.merge(demand_df, left_index=True, right_index=True, how=how)
    return merged.sort_index()


def merge_driver(df: pd.DataFrame, driver_csv, value_cols, datetime_col: str = "StartTime",
                 how: str = "left") -> pd.DataFrame:
    """Stage 2 extension point: merge an additional fundamental-driver series.

    e.g. wind generation, gas price, carbon price, interconnector flow. Kept generic so
    new drivers can be added without touching the forecasting code.
    """
    d = pd.read_csv(driver_csv, parse_dates=[datetime_col]).set_index(datetime_col)
    return df.merge(d[value_cols], left_index=True, right_index=True, how=how).sort_index()


def assemble_market_data(price_glob: str = "SystemSellAndBuyPrices-2017*.csv",
                         demand_csv: str | None = None) -> pd.DataFrame:
    """Build the base market frame: prices + demand, datetime-indexed."""
    price_files = glob.glob(str(PRICE_DIR / price_glob))
    prices = merge_price_csvs(price_files)
    if demand_csv is None:
        candidates = sorted(DEMAND_DIR.glob("RollingSystemDemand-*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No demand CSV found under {DEMAND_DIR} (see README)."
            )
        demand_csv = candidates[0]
    return merge_with_demand(prices, demand_csv)
