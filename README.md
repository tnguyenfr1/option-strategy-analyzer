# 📊 Option Strategy Analyzer

Multi-leg option payoff analyzer with live market data, probability analysis,
roll & assignment simulation, and per-contract price history.

**Created by [Thuan](https://github.com/YOUR_GITHUB_USERNAME) in collaboration
with Claude (Anthropic).**

## Features

- **Live option chains** (yfinance) — real strikes, expirations, bid/ask/mid,
  volume, open interest; auto-sync on every contract change
- **P&L chart** with filled profit/loss regions, adjustable analyze date
  (0 → expiration), and breakevens at any DTE
- **Probability overlay** — lognormal price distribution with ±1σ/±2σ implied
  move markers, Probability of Profit, P(max profit), P(max loss)
- **IV computed from the chain mid** by inverting Black-Scholes (matches
  OptionCharts/OptionStrat methodology), with yfinance IV as fallback
- **What-If Simulator** — close-now vs. hold-to-expiration P&L at any
  price/date, per-leg breakdown, assignment scenarios
- **Roll Planner** — pick a leg, choose a new strike/expiration from the live
  chain, see roll credit/debit and updated position metrics
- **Option price history** — real contract candles + theoretical BS price +
  underlying overlay, like a per-contract chart on OptionCharts
- **Price × Date P&L table** — color-coded outcome grid

## Run locally

```bash
pip install -r requirements.txt
streamlit run option_strategy_analyzer.py
```

## Disclaimer

Theoretical values from Black-Scholes; probabilities from a risk-neutral
lognormal model. For educational purposes only — not financial advice.

## Credits

- Built by **Thuan** with **Claude** (Anthropic)
- Inspired by [optioncharts.io](https://optioncharts.io) and
  [optionstrat.com](https://optionstrat.com)
- Market data via [yfinance](https://github.com/ranaroussi/yfinance)
