# Battery Energy Storage Trading & Arbitrage Strategy

A quantitative trading strategy for short-term UK electricity markets: forecast
short-term price/imbalance opportunities, convert probabilistic forecasts into battery
charge/discharge decisions under realistic operating constraints, and evaluate the
**risk-adjusted commercial outcome** — P&L, £/MWh, Sharpe, drawdown, VaR — against
benchmarks.

The guiding question is *not* "how accurate is the price forecast?" but:
**does the forecast produce a commercially useful signal after constraints, costs and
risk?** See [ROADMAP.md](ROADMAP.md) for the five-stage plan and what is / isn't built.

## Pipeline

```
market data → features → forecast → decision → simulate (constraints/costs) → risk-adjusted P&L → benchmarks
```

| Stage | Module | Responsibility |
|------|--------|----------------|
| 0 | `config.py`, `data_pipeline.py` | typed config; load & merge price + demand CSVs |
| 2 | `features.py` | past-only feature engineering (+ driver hook) |
| 1 | `labels.py`, `forecasting.py` | forward-looking targets; time-series-CV models |
| 3 | `decision.py`, `strategy.py`, `simulate.py` | forecast → trade → constrained dispatch |
| 4/5 | `evaluate.py`, `benchmarks.py` | Sharpe/drawdown/VaR; strategy comparison |
| — | `backtest.py` | orchestrates a leakage-free, **test-window-only** backtest |

### Leakage control (the key correctness property)
Every feature uses only information available at or before time *t*; the only
forward-looking quantity is the **label**. The backtest splits **chronologically** and
evaluates strictly on the later, unseen window — unlike the original code, which
predicted over the whole dataset. This is verified in `tests/test_pipeline.py`
(`test_features_have_no_lookahead`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Market data (not committed)
Fetch it directly from the **Elexon BMRS Insights API** (free, keyless):

```bash
python fetch_bmrs.py --from 2024-01-01 --to 2024-03-31
```

This writes the CSVs to the expected layout:

```
SystemSellAndBuyPrices/SystemSellAndBuyPrices-<range>.csv  # StartTime, SystemSellPrice, SystemBuyPrice (£/MWh, DISEBSP)
RollingSystemDemand/RollingSystemDemand-<range>.csv        # StartTime, Demand (MW, ITSDO from demand/outturn)
```

Note: UK imbalance prices go **negative and zero**, so features avoid log returns
(volatility is the trailing std of price changes).

## Run

```bash
python training.py     # train + persist forecast models on the training window
python backtest.py     # honest out-of-sample backtest + benchmark comparison
pytest -q              # verify the pipeline on synthetic data (no CSVs needed)
```

`backtest.py` prints a strategy comparison table (ML vs a perfect-foresight *myopic*
reference — not a true optimal-dispatch bound — vs
naïve baseline) with P&L, £/MWh, Sharpe, max drawdown and VaR, plus the top forecast
drivers.

## Configuration
- `BatteryConfig.yaml` — capacity/power/efficiency/SoC limits/degradation (consistent kWh & kW).
- `TradingConfig.yaml` — starting balance, transaction cost, and `min_edge_per_mwh`
  (the commercial gate: forecasts below this expected edge are not traded).

## Status & honesty note
No performance figures are quoted here on purpose — the real CSVs aren't in the repo, so
every number would be synthetic. The evaluation harness computes metrics from whatever
data is supplied; run `backtest.py` on real UK data to produce reportable results.
The legacy 40 MB `refined_model1_*.joblib` predates this refactor and is incompatible
with the current features — retrain with `training.py`.
