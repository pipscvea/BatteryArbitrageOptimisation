# Battery Energy Storage Trading & Arbitrage Strategy

A quantitative trading strategy for short-term UK electricity markets: forecast
short-term price/imbalance opportunities, convert probabilistic forecasts into battery
charge/discharge decisions under realistic operating constraints, and evaluate the
**risk-adjusted commercial outcome** — P&L, £/MWh, Sharpe, drawdown, VaR — against
benchmarks.

The guiding question is *not* "how accurate is the price forecast?" but:
**does the forecast produce a commercially useful signal after constraints, costs and
risk?**

📊 **[RESULTS.md](RESULTS.md)** — headline results, findings and figures (start here).
🗺️ **[ROADMAP.md](ROADMAP.md)** — the five-stage plan and what is / isn't built.

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

## Data & forecast

**Data** — all from the free, keyless [Elexon BMRS Insights API](https://bmrs.elexon.co.uk/)
(`fetch_bmrs.py`); headline runs use **2 years, 2023–2024, half-hourly** (~35,000 settlement
periods):

| Signal | BMRS dataset | What it is |
|---|---|---|
| System Sell / Buy Price | DISEBSP | GB electricity **imbalance (cash-out) prices**, £/MWh — the traded price |
| Demand | ITSDO (`demand/outturn`) | Transmission system demand, MW |
| Wind | FUELINST (`generation/outturn/summary`) | Wind generation, MW |
| Interconnector flow | FUELINST | Net flow across all interconnectors (INTFR, INTNED, …), MW |
| Gas | FUELINST | CCGT generation, MW |

From these, `features.py` builds **backward-only** features: price lags / rolling
mean & std / spread / volatility, demand lags & rolling stats, calendar (hour, day-of-week,
month, weekend), and wind level/lag/moving-average/**ramp** + interconnector + gas.

**What the model forecasts** — the label is the *only* forward-looking quantity (which is
what keeps the backtest leak-free). Three targets, all predicted from past-only features:

1. **Forward price change** — `SystemSellPrice(t+h) − SystemSellPrice(t)` over a tuned
   horizon (default **h = 4**, ~2 h ahead). A random forest predicts this; the decision
   layer turns it into an expected arbitrage **edge**. *This is the target that drives trading.*
2. **P(tradeable move)** — a classifier estimating "does the next period offer a spread
   large enough to justify a trade?" (`predict_proba`, ≈0.78 AUC out-of-sample).
3. **Forward price *path*** — a multi-output regressor over the next 48 periods, used only
   to feed the LP/MPC dispatch.

The model forecasts **short-term price/imbalance opportunities — not the battery actions**.
A separate decision/optimisation layer converts the forecast into charge/discharge subject
to the operating constraints. Success is measured as **commercial value after costs and
risk**, not forecast accuracy (see the anti-goal in [ROADMAP.md](ROADMAP.md)).

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
Drivers/Drivers-<range>.csv                                # StartTime, Wind, InterconnectorNet, Gas (MW, FUELINST @30min)
```

Stage 2 driver features (wind + lags/ramp, net interconnector flow, gas/CCGT) are merged
and engineered automatically when the `Drivers/` CSVs are present, and silently skipped
otherwise. Wind is the dominant short-term price driver (negatively correlated with price).

Note: UK imbalance prices go **negative and zero**, so features avoid log returns
(volatility is the trailing std of price changes).

## Run

```bash
python training.py     # train + persist forecast models on the training window
python backtest.py     # single-window tuned out-of-sample backtest + benchmark comparison
python robustness.py   # expanding-window walk-forward across every quarter (+ plot in figs/)
python risk.py         # Stage 5 risk report: tail metrics, forecast-error sensitivity, regime stress
python docs/build_report.py  # render the "How it works" PDF from figs/ into docs/
pytest -q              # verify the pipeline on synthetic data (no CSVs needed)
```

`robustness.py` is the credibility run: it walks forward quarter by quarter (train on all
prior data, test on the quarter) and reports per-quarter and aggregate P&L / Sharpe /
drawdown / capture-vs-benchmark, saving a plot to `figs/robustness_walkforward.png`.

`backtest.py` prints a strategy comparison table — ML vs the **LP optimum** (perfect-
foresight optimal dispatch, `optimize.py`, a genuine upper bound), the perfect-foresight
*myopic* reference, and the naïve baseline — with P&L, £/MWh, Sharpe, max drawdown and
VaR, plus the top forecast drivers. Capture-vs-LP is the honest "how much of the
achievable value did the forecast get?" metric.

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
