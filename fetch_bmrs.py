"""Fetch UK market data from the Elexon BMRS Insights API into the layout the pipeline
expects. Keyless, stdlib-only.

Datasets:
  * DISEBSP  — settlement System Sell/Buy Prices (per settlement date)
                -> SystemSellAndBuyPrices/SystemSellAndBuyPrices-<from>_<to>.csv
  * INDO/ITSDO (demand/outturn) — half-hourly demand outturn (chunked date range)
                -> RollingSystemDemand/RollingSystemDemand-<from>_<to>.csv
    We use ITSDO (transmission system demand, MW) as ``Demand``.

Usage:
    python fetch_bmrs.py --from 2023-01-01 --to 2023-12-31
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # certifi recommended; fall back to system trust store
    _SSL_CTX = ssl.create_default_context()

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
ROOT = Path(__file__).resolve().parent


def _get(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=_SSL_CTX) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry with backoff
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_system_prices(start: date, end: date) -> list[dict]:
    rows: dict[str, dict] = {}
    for d in _daterange(start, end):
        url = f"{BASE}/balancing/settlement/system-prices/{d.isoformat()}?format=json"
        for rec in _get(url).get("data", []):
            if rec.get("systemSellPrice") is None:
                continue
            rows[rec["startTime"]] = {
                "StartTime": rec["startTime"],
                "SystemSellPrice": rec["systemSellPrice"],
                "SystemBuyPrice": rec["systemBuyPrice"],
            }
        time.sleep(0.1)  # be polite
    return [rows[k] for k in sorted(rows)]


def fetch_demand(start: date, end: date, chunk_days: int = 14) -> list[dict]:
    rows: dict[str, dict] = {}
    d = start
    while d <= end:
        chunk_end = min(d + timedelta(days=chunk_days - 1), end)
        url = (f"{BASE}/demand/outturn?settlementDateFrom={d.isoformat()}"
               f"&settlementDateTo={chunk_end.isoformat()}&format=json")
        for rec in _get(url).get("data", []):
            demand = rec.get("initialTransmissionSystemDemandOutturn")
            if demand is None:
                continue
            rows[rec["startTime"]] = {"StartTime": rec["startTime"], "Demand": demand}
        d = chunk_end + timedelta(days=1)
        time.sleep(0.1)
    return [rows[k] for k in sorted(rows)]


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):>6} rows -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="frm", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="to", required=True, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()
    start = datetime.strptime(args.frm, "%Y-%m-%d").date()
    end = datetime.strptime(args.to, "%Y-%m-%d").date()
    tag = f"{start.isoformat()}_{end.isoformat()}"

    print(f"Fetching system prices {tag} ...")
    prices = fetch_system_prices(start, end)
    _write_csv(prices, ROOT / "SystemSellAndBuyPrices" / f"SystemSellAndBuyPrices-{tag}.csv",
               ["StartTime", "SystemSellPrice", "SystemBuyPrice"])

    print(f"Fetching demand {tag} ...")
    demand = fetch_demand(start, end)
    _write_csv(demand, ROOT / "RollingSystemDemand" / f"RollingSystemDemand-{tag}.csv",
               ["StartTime", "Demand"])


if __name__ == "__main__":
    main()
