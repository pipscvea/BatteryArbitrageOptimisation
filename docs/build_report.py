"""Generate the 'How it works' PDF report (docs/BatteryArbitrage_HowItWorks.pdf).

Standalone documentation generator — reads the result figures from ``figs/`` and writes a
self-contained PDF. It does NOT run the pipeline; regenerate the figures first with
``python robustness.py`` and ``python risk.py`` if you want the latest numbers/plots.

Requires reportlab (see requirements.txt). Run from anywhere:

    python docs/build_report.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
OUT = ROOT / "docs" / "BatteryArbitrage_HowItWorks.pdf"

INK = colors.HexColor("#14201d")
TEAL = colors.HexColor("#0b6b5f")
MUTED = colors.HexColor("#5a6a64")
LINE = colors.HexColor("#dce3df")
SOFT = colors.HexColor("#eef3f1")

ss = getSampleStyleSheet()
def style(name, **kw):
    parent = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=parent, **kw)

BODY = style("body", fontName="Helvetica", fontSize=10, leading=15, textColor=INK, spaceAfter=8)
LEDE = style("lede", parent=BODY, textColor=MUTED, fontSize=10.5, leading=15.5)
H1 = style("h1", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=INK, spaceBefore=6, spaceAfter=4)
EYEBROW = style("eyebrow", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=TEAL, spaceAfter=2)
H2 = style("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=INK, spaceBefore=10, spaceAfter=3)
CAP = style("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=4, spaceAfter=6)
BULLET = style("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CODE = style("code", fontName="Courier", fontSize=8.5, leading=12, textColor=INK, backColor=SOFT,
             borderPadding=6, spaceBefore=4, spaceAfter=8)
COVER_T = style("ct", fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=INK)
COVER_S = style("cs", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED)

story = []

def section(eyebrow, title):
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=6))
    story.append(Paragraph(eyebrow.upper(), EYEBROW))
    story.append(Paragraph(title, H1))
    story.append(Spacer(1, 3))

def para(t, s=BODY): story.append(Paragraph(t, s))
def bullets(items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="•"))
    story.append(Spacer(1, 4))

def table(data, colwidths, header=True, right_cols=()):
    t = Table(data, colWidths=colwidths, hAlign="LEFT")
    cmds = [
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("TEXTCOLOR", (0,0), (-1,-1), INK),
        ("LINEBELOW", (0,0), (-1,-1), 0.5, LINE),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]
    if header:
        cmds += [("FONT", (0,0), (-1,0), "Helvetica-Bold", 8.5),
                 ("TEXTCOLOR", (0,0), (-1,0), MUTED),
                 ("BACKGROUND", (0,0), (-1,0), SOFT),
                 ("LINEBELOW", (0,0), (-1,0), 1, TEAL)]
    for c in right_cols:
        cmds.append(("ALIGN", (c,0), (c,-1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    story.append(t); story.append(Spacer(1, 8))

# ---------------- COVER ----------------
story.append(Spacer(1, 5*cm))
story.append(Paragraph("UK POWER MARKETS · QUANTITATIVE TRADING", EYEBROW))
story.append(Spacer(1, 6))
story.append(Paragraph("Battery Storage Trading &amp; Arbitrage", COVER_T))
story.append(Spacer(1, 10))
story.append(Paragraph("How the program works — a technical explainer of the data, "
                       "forecasting, decision logic, optimisation, and evaluation.", COVER_S))
story.append(Spacer(1, 1.2*cm))
story.append(HRFlowable(width="40%", thickness=2, color=TEAL, hAlign="LEFT"))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph("A Python/scikit-learn pipeline that forecasts short-term UK electricity "
                       "imbalance opportunities and converts them into battery charge/discharge "
                       "decisions under realistic operating constraints, then measures the "
                       "risk-adjusted commercial outcome.", LEDE))
story.append(PageBreak())

# ---------------- 1. WHAT IT DOES ----------------
section("Overview", "What the program does")
para("The program is a quantitative trading strategy for a grid-connected battery operating "
     "against Great Britain's electricity <b>imbalance (cash-out) market</b>. It buys (charges) "
     "when power is expected to be cheap and sells (discharges) when it is expected to be dear, "
     "subject to the battery's physical limits and trading costs.")
para("The organising principle is deliberately commercial, not statistical:", LEDE)
para("<i>“Does the forecast generate a commercially useful trading signal after constraints, "
     "costs and risk — not how accurate is the price forecast?”</i>")
para("Everything downstream (the decision rule, the optimiser, the risk analysis) exists to "
     "answer that question honestly. The end-to-end flow is:")
story.append(Paragraph("market data &rarr; features &rarr; forecast &rarr; decision &rarr; "
             "simulate (constraints/costs) &rarr; optimise (LP/MPC) &rarr; P&amp;L &rarr; "
             "walk-forward robustness &rarr; risk analysis", CODE))

# ---------------- 2. DATA & FORECAST ----------------
section("Inputs", "Data &amp; what the model forecasts")
para("All data comes from the free, keyless <b>Elexon BMRS Insights API</b> (fetched by "
     "<font face='Courier'>fetch_bmrs.py</font>). Headline runs use two years of half-hourly "
     "data (2023–2024, ~35,000 settlement periods).")
table([
    ["Signal", "BMRS dataset", "What it is"],
    ["System Sell / Buy Price", "DISEBSP", "GB imbalance prices, £/MWh — the traded price"],
    ["Demand", "ITSDO", "Transmission system demand, MW"],
    ["Wind", "FUELINST", "Wind generation, MW"],
    ["Interconnector flow", "FUELINST", "Net flow across all interconnectors, MW"],
    ["Gas", "FUELINST", "CCGT generation, MW"],
], [4.6*cm, 3.0*cm, 8.4*cm])
para("From these, <font face='Courier'>features.py</font> builds <b>backward-only</b> features "
     "(price lags, rolling mean/std, spread, volatility; demand lags; calendar; and wind "
     "level/lag/moving-average/ramp, interconnector, gas). The model then predicts three "
     "<b>forward-looking</b> targets — the only place future information is used, which is "
     "what keeps the backtest leakage-free:")
bullets([
    "<b>Forward price change</b> over a tuned horizon (default 4 periods / ~2h) — the target "
    "that drives trading, turned into an expected arbitrage <b>edge</b>.",
    "<b>P(tradeable move)</b> — a classifier estimating the probability the next period offers "
    "a spread worth trading (~0.78 AUC out-of-sample).",
    "<b>Forward price path</b> over 48 periods — a multi-output model that feeds the LP/MPC dispatch.",
])
para("Crucially, the model forecasts <b>opportunities, not battery actions</b> — a separate "
     "decision/optimisation layer converts forecasts into dispatch.", LEDE)

# ---------------- 3. ARCHITECTURE ----------------
section("Architecture", "How the code is organised")
para("The pipeline is decomposed into small, independently testable modules. Each stage can be "
     "swapped or inspected without touching the others.")
table([
    ["Module", "Responsibility"],
    ["config.py", "Typed battery &amp; trading configuration (capacity, power, efficiency, SoC limits, costs)."],
    ["fetch_bmrs.py", "Pull prices, demand and generation drivers from the BMRS API into CSVs."],
    ["data_pipeline.py", "Merge prices + demand + drivers into one datetime-indexed frame."],
    ["features.py", "Backward-only feature engineering (+ driver features when present)."],
    ["labels.py", "Forward-looking targets: price change, tradeable-move, price path."],
    ["forecasting.py", "Train the random-forest forecasters (time-series CV)."],
    ["decision.py", "Convert a forecast into an expected edge and a sized trade request."],
    ["simulate.py", "Battery simulator enforcing all physical + cost constraints; equity curve."],
    ["optimize.py", "Perfect-foresight LP dispatch (HiGHS) — the true upper bound."],
    ["mpc.py", "Receding-horizon MPC (forecast + rolling LP), with a robust prior knob."],
    ["evaluate.py", "P&amp;L, £/MWh, Sharpe, drawdown, VaR."],
    ["benchmarks.py", "Perfect-foresight-myopic reference and naive time-of-day baseline."],
    ["backtest.py", "Orchestrates a leakage-free, tuned, test-window-only backtest."],
    ["robustness.py", "Expanding-window walk-forward across every quarter."],
    ["risk.py", "Tail metrics, forecast-error sensitivity, regime stress."],
], [3.3*cm, 12.7*cm])

# ---------------- 4. HOW A RUN WORKS ----------------
section("Mechanics", "How a run works")
para("<b>1. Assemble &amp; engineer.</b> Prices, demand and drivers are merged and turned into "
     "features. <b>2. Split chronologically</b> into train / validation / test — no shuffling. "
     "<b>3. Tune on validation</b> (forecast horizon and trade sizing) so the test window is never "
     "used for any decision. <b>4. Fit</b> the forecaster on train+validation. <b>5. Decide &amp; "
     "simulate</b> on the unseen test window only. <b>6. Compare</b> against the LP optimum, the "
     "myopic reference and the naive baseline.")
para("<b>Leakage control</b> is the key correctness property: every feature uses only information "
     "available at or before time <i>t</i>; the only forward-looking quantity is the label; and the "
     "split is strictly chronological. This is enforced by an automated test that perturbs a future "
     "price and asserts no earlier feature changes.")
para("<b>The decision layer</b> converts the predicted price change into a round-trip edge "
     "(£/MWh, net of efficiency loss and costs) and only trades when that edge clears a "
     "commercial gate — so a forecast that is directionally right but too small to beat costs "
     "correctly does nothing. <b>The simulator</b> then enforces capacity, min/max state-of-charge, "
     "per-period power, one-way efficiency, transaction and degradation costs, and marks the stored "
     "energy to market to produce an equity curve.")

story.append(PageBreak())

# ---------------- 5. RESULTS ----------------
section("Results", "Does it work? (2 years, out-of-sample)")
para("<b>Walk-forward robustness.</b> For each calendar quarter the model is trained on all prior "
     "data and evaluated out-of-sample on that quarter. The strategy is profitable in 7/7 quarters, "
     "beats the naive baseline in 6/7, and captures ~56% of the LP optimum (the true upper bound).")
if (FIGS / "robustness_walkforward.png").exists():
    img = Image(str(FIGS / "robustness_walkforward.png"))
    img._restrictSize(16*cm, 6*cm); img.hAlign = "LEFT"
    story.append(img)
    story.append(Paragraph("Out-of-sample P&amp;L by quarter (LP optimum &gt; myopic &gt; ML &gt; naive) "
                 "and cumulative P&amp;L across 2023–2024. The flat spot is the weak 2024Q1 regime.", CAP))

para("<b>Strategy comparison</b> on a single tuned test window:", H2)
table([
    ["Strategy", "P&L (£)", "Sharpe", "% of LP"],
    ["LP optimum (upper bound)", "48,013", "24.8", "100%"],
    ["Perfect-foresight myopic (reference)", "37,120", "19.6", "77%"],
    ["ML forecast (deployed heuristic)", "30,057", "16.3", "63%"],
    ["MPC (forecast + rolling LP)", "9,907", "5.7", "21%"],
    ["Naive time-of-day (baseline)", "9,346", "3.9", "19%"],
], [8.0*cm, 3.0*cm, 2.5*cm, 2.5*cm], right_cols=(1,2,3))

para("<b>Risk profile.</b> Beyond P&amp;L: tail risk, drawdown, and how sensitive returns are to "
     "forecast error. Returns are strongly right-skewed — the money is made in price spikes "
     "(15% of periods, ~64% of realised P&amp;L).")
if (FIGS / "risk_analysis.png").exists():
    img2 = Image(str(FIGS / "risk_analysis.png"))
    img2._restrictSize(16*cm, 5*cm); img2.hAlign = "LEFT"
    story.append(img2)
    story.append(Paragraph("P&amp;L distribution (fat right tail, VaR marked); drawdown from peak; and "
                 "forecast-error sensitivity — P&amp;L decays to zero at ~£80/MWh of injected "
                 "forecast noise.", CAP))
table([
    ["Sharpe / Sortino", "VaR / CVaR (95%)", "Max drawdown", "Exposure"],
    ["~16 / ~21", "£37 / £69", "0.6%", "trades 87% of periods"],
], [4.0*cm, 4.3*cm, 3.3*cm, 4.4*cm])

# ---------------- 6. FINDINGS ----------------
section("Insights", "What matters more than the P&amp;L")
bullets([
    "<b>Complexity does not automatically create value.</b> The most sophisticated policy (MPC) "
    "underperformed the simple edge-heuristic: feeding a point forecast into an optimiser makes it "
    "over-commit to trajectories that turn out wrong. A robust variant did not recover the gap.",
    "<b>Forecast accuracy is not P&amp;L.</b> The sensitivity curve monetises exactly how much the "
    "strategy leans on forecast quality — the binding constraint is a better forecast, not a "
    "cleverer optimiser.",
    "<b>The money is in the spikes.</b> A small fraction of high-price periods drives most of the return.",
    "<b>Read realised cash, not mark-to-market.</b> Negative-price periods looked like a loss but were "
    "actually correct behaviour (charging when paid to consume) — the loss was transient inventory "
    "revaluation.",
])

# ---------------- 7. LIMITATIONS ----------------
section("Scope", "Limitations &amp; next steps")
para("<b>Limitations:</b> one battery configuration (£ figures scale with size); two years of one "
     "market; a heuristic policy driven by a point forecast; and a backtest — no live data latency, "
     "execution or settlement. <b>Next steps:</b> a calibrated probabilistic forecast with "
     "confidence-based sizing; a chance-constrained / scenario-based stochastic optimiser; richer "
     "drivers (solar, weather, outages) with SHAP attribution; and a VaR term structure with "
     "extreme-scenario replays.")
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=6))
para("Reproducible end to end: <font face='Courier'>fetch_bmrs.py</font> &rarr; "
     "<font face='Courier'>backtest.py</font> / <font face='Courier'>robustness.py</font> / "
     "<font face='Courier'>risk.py</font>. Pipeline verified on synthetic data in "
     "<font face='Courier'>tests/</font>. All figures from real UK data — nothing manufactured.", CAP)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2*cm, 1.1*cm, "Battery Storage Trading & Arbitrage - How it works")
    canvas.drawRightString(A4[0]-2*cm, 1.1*cm, "%d" % doc.page)
    canvas.setStrokeColor(LINE); canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.restoreState()


def main():
    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title="Battery Storage Trading & Arbitrage - How it works",
                            author="pipscvea")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
