# Roadmap — Battery Energy Storage Trading & Arbitrage Strategy

**Objective (the thing we optimise the whole project around):**
> Does the forecast generate a *commercially useful trading signal* after constraints,
> costs and risk are considered? — not "how statistically accurate is the price forecast".

End-to-end target pipeline:

```
Market data → fundamental analysis → probabilistic forecast → opportunity detection
   → trading decision → optimisation → P&L → risk analysis → explanation of performance
```

## Five stages

### 1. Market forecasting
Forecast something *commercially meaningful*, not just "next price":
- intraday price, day-ahead → intraday spread, imbalance price
- short-term price direction, volatility
- **probability of a sufficiently large price move** (the trading-oriented target):
  "Given information at 12:00, P(next period offers a spread large enough to justify a trade)?"

Status: `labels.py` defines forward-looking targets incl. a probabilistic
`tradeable_move` classifier target. Regression + directional targets implemented.
Probabilistic calibration is the priority extension.

### 2. Identify market drivers
Make it Market Analysis & Trading, not generic ML. Candidate drivers:
demand, wind/solar generation, interconnector flows, system imbalance, gas price,
carbon, temperature/weather, outages, historical price behaviour.
Then ask: *why* did the model predict this move? (feature attribution).

Status: `features.py` builds price/demand/time features today and exposes a documented
extension point (`add_driver_features`) for the fundamental drivers above once their
data is sourced. Attribution hook noted in `forecasting.py`.

### 3. Convert forecast → trading decision
Not `forecast = £X/MWh`, but `forecast distribution → expected opportunity → optimal action`,
subject to: capacity, SoC, charge/discharge efficiency, max power, degradation cost,
transaction cost, min/max SoC, and (ideally) forecast uncertainty.

Status: `decision.py` (heuristic edge-vs-cost policy) + `simulate.py` (enforces all
listed constraints). Optimisation-based dispatch (LP/MPC) is a planned alternative policy.

### 4. Measure the commercial outcome
Aim to be able to say: "Backtested across X months, £X/MWh average gross value,
Sharpe X, max drawdown X%, Y% over benchmark." **Do not manufacture numbers.**
Compare strategies: naïve forecast, statistical, ML, optimisation, benchmark battery.

Status: `evaluate.py` computes P&L, £/MWh, Sharpe, drawdown, hit rate.
`benchmarks.py` provides perfect-foresight upper bound + naïve rule baselines.

### 5. Risk analysis
Beyond P&L: volatility, max drawdown, downside risk, exposure, VaR, stress scenarios,
sensitivity to forecast errors, performance in extreme volatility.

Status: `evaluate.py` computes historical VaR + drawdown. Stress scenarios and
forecast-error sensitivity are planned (`risk.py`, not yet built).

## What is deliberately NOT done yet
- No live/real performance numbers — the CSVs (`SystemSellAndBuyPrices/`,
  `RollingSystemDemand/`) are not in the repo. Everything is verified on synthetic data.
- Fundamental driver data (wind/solar/gas/carbon/weather) not yet sourced → Stage 2 hooks only.
- Probabilistic forecast calibration, optimisation-based dispatch, VaR stress suite: planned.

## Anti-goal
Not "I trained XGBoost/LSTM/Transformer to predict prices with 97% accuracy."
Statistical accuracy ≠ money. Optimise for commercial value after costs and risk.
