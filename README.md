# Quantitative Strategies

Welcome, and thank you for stopping by. This repository holds the research behind my quantitative trading portfolio: four connected Python projects, from regression signals and time-series features to a deep Q-network (DQN) stock-trading agent and an options agent that trades calls and puts it prices itself, plus a risk-parity backtest.

You can explore all of the results, with interactive charts and a live trading desk, at **[damionm.io](https://damionm.io)**.

## About me

I'm Damion Marshall, a Columbia University graduate (BA, Mathematics–Statistics) seeking quantitative trading and research roles. I enjoy building strategies end to end: sourcing the data, designing a fair test, and being honest about what the results do and don't show. If you'd like to talk about any of this work, I'd be glad to hear from you at [contact@damionm.io](mailto:contact@damionm.io).

## How the projects fit together

Each project asks a harder question than the last, building on what came before:

| # | Project | Question | Notebook |
|---|---------|----------|----------|
| 01 | Regression models | Can simple models predict tomorrow's direction across the S&P 500? | `regression_models_comparison.ipynb` |
| 02 | Time-series features | What does one stock's own history reveal? | `time_series.ipynb` |
| 03 | RL trading agent | Can an agent learn when to buy and sell from those features? | `rl_trading_agent.ipynb` |
| 04 | GBM options RL agent | Can it trade options that it prices itself? | `gbm_options_rl_agent.ipynb` |
| + | Risk parity | How should risk be balanced across asset classes? | `risk_parity_backtest.ipynb` |

Every notebook draws on `func_lib.py`, a shared library for data handling, chronological splits, backtests, performance statistics, the DQN agent and its episode runners, option pricing and volatility models (including a small `GBM` class), and a common chart theme. Its core routines have been checked independently against reference implementations.

## What's new

- **Portfolio website.** The full write-up now lives at [damionm.io](https://damionm.io), with a page for each project, a Notes page, and a "How I tested this" summary for every result.
- **Live trading desk.** The trained DQN agent runs in your browser at [damionm.io/desk](https://damionm.io/desk). You can replay each options episode, compare episodes side by side, and see every trade.
- **Refreshed options results.** The GBM options agent was retrained, and the results below reflect the latest clean run, including per-episode results and updated benchmarks.
- **Refreshed charts.** Every project chart has been redrawn in a consistent graphite theme.
- **A `GBM` class.** `func_lib.py` now has a small `GBM` class that fits μ and σ, gives a forecast range, simulates paths and values options, built on the same tested pricing functions.
- **Clearer names.** In the options notebook, `rf` is always the risk-free rate, `log_ret` the log returns and `features` the daily feature table.
- **Simpler repository.** The HTML slide decks have been retired in favour of the website, and the regression work is consolidated into a single notebook.

---

## 01 · Regression models

Five models forecast the S&P 500's next-day move from 1-, 5-, 15- and 20-day momentum, using data from 2000 onward. Each forecast becomes a long/flat strategy, compared with an equal-weighted benchmark.

- **Models:** linear regression (OLS), Lasso (alpha chosen by cross-validation), logistic regression, an unpruned decision tree, and a decision tree tuned by grid search.
- **Method:** one chronological 70/30 split. Features are scaled on training data only, and cross-validation folds are cut by date, so no model sees the future.
- **Results (out of sample, from September 2018):** no model beat the benchmark. Hit rates were 50–52% and ROC-AUC was 0.50–0.51. Lasso came closest, growing $1 to about $3.27 against the benchmark's $3.45, largely because it stayed long 90% of the time.
- **Caveats:** results exclude trading costs and use today's S&P 500 members, which flatters every strategy slightly.

## 02 · Time-series features

Builds and charts AAPL features from March 2020 to August 2024: 20/50-day moving averages, Bollinger Bands, RSI, price and return volatility, and volume.

- AAPL rose about 240% and closed above its 50-day average on 62% of days.
- RSI was above 70 on 11% of days and below 30 on 1%, typical of a long uptrend.
- Saves `AAPL_trading_features.csv`, which supplies the seven features the RL agent sees.

## 03 · RL trading agent (DQN)

A deep Q-network trades AAPL with three actions: hold, buy one share (up to five), and sell the oldest share.

- **State:** daily changes in the seven features, plus the agent's own position.
- **Training:** the reward is daily mark-to-market P&L, with 5 bp costs, action masking, a target network and a paced exploration schedule. It trains on March 2020 to April 2023 and tests on the following 338 unseen days.
- **Results:**

  | Policy | Test P&L | Max drawdown |
  |---|---|---|
  | Episode 2 model (best checkpoint) | $301 | — |
  | Final model | $130 net, $162 gross (170 trades) | $64 (smallest) |
  | Buy-and-hold | $308 | $163 |
  | Random | $241 | $119 |

  The final model took the least risk but earned less than both benchmarks. The episode 2 checkpoint mostly held its position and nearly matched buy-and-hold, a useful reminder that the best model is not always the last one trained.
- **Includes:** a reusable backtest function, a full trade log and post-trade analytics.

## 04 · GBM options RL agent

Extends the agent to options on SPY, using VIX as the implied-volatility proxy:

1. **Price model:** geometric Brownian motion, fitted by maximum likelihood, with diagnostics showing where fat tails break it.
2. **Volatility forecasts:** historical, EWMA, GARCH(1,1), HAR-RV and VIX, each refitted walk-forward and scored on QLIKE.
3. **Pricing engine:** Monte Carlo with antithetic variates, checked against Black-Scholes, plus the Greeks. A `GBM` class wraps the engine for one set of parameters:

   ```python
   from func_lib import GBM
   model = GBM.fit(log_returns)               # maximum-likelihood mu and sigma
   model.forecast(S0=500, T=63/252)           # expected price, median and 90% range
   paths = model.simulate(500, 63/252, 63, 20_000, seed=42)
   model.option_value(S=500, K=500, T=21/252, r=0.04, kind='put')
   ```

   The daily walk-forward loop still calls the vectorised functions directly, since μ and σ change every day.
4. **Edge signal:** the GBM expected value of an option compared with its market price.
5. **DQN agent:** six actions (hold, buy call, buy put, sell call, sell put, close), with bid/ask costs, Reg-T-style margin on short options, forced buy-backs and a crash filter.

**Results (out of sample from October 2021, $100k account):**

| Strategy | Return | Sharpe | Max drawdown |
|---|---|---|---|
| DQN agent (episode 3) | +59% | 0.74 | 20% |
| SPY buy-and-hold | +80% | 0.76 | 25% |
| GBM-edge rule | −1.6% | — | — |
| Random policy | −39% | — | — |

- The agent placed 168 trades and learned on its own to sell puts, collecting the variance risk premium: implied volatility exceeded the volatility that followed on 85% of days, by 3.8 points on average.
- It came close to SPY's risk-adjusted return with a shallower drawdown, while the pricing-model rule alone and random trading both lost money.
- Every trade is logged by position type (long or short, call or put). Full design notes are in `gbm-options-rl-agent-design.md`.

## + · Risk parity

Inverse-volatility weights across S&P 500 e-mini, gold and 10-year Treasury note futures (2015–2025), compared with an equal-weight portfolio. Weights use only the previous day's data, and the rebalance frequency and trading costs are configurable.

- Risk parity grew $1 to about $1.61, against about $2.36 for equal weight, since it leaned toward lower-volatility bonds during a strong decade for stocks.
- Both portfolios drew down about 16% in 2022, when bonds fell alongside stocks.

---

## Current project: anomaly detection

I'm currently building a walk-forward research environment for detecting market anomalies and regime changes. It is a work in progress and isn't in this repository yet; results will follow once the testing is complete.

## Key takeaways

- Daily direction is close to unpredictable from simple momentum features, so time in the market drove most of the regression results.
- The environment matters as much as the network. Position-aware states, the right reward and paced exploration were what made the agents trade sensibly.
- The best checkpoint was not always the final one, so a separate validation period should guide model selection.
- Option sellers collect a real variance risk premium, but margin and crash controls decide whether a strategy survives.

## Getting started

```bash
pip install numpy pandas scipy scikit-learn statsmodels plotly yfinance tensorflow keras nbformat
```

1. Keep `func_lib.py` in the same folder as the notebooks, or in the parent folder.
2. Run `time_series.ipynb` before `rl_trading_agent.ipynb`; it writes the agent's input CSV.
3. The other notebooks download their own data with `yfinance` and need internet access.
4. Each notebook's settings are in its **Configuration** cell, and every code cell opens with a short explanation of what it does.
5. For reproducible results, use **Restart & Run All** so each agent trains from a fresh start.

## Repository layout

```
func_lib.py                          shared helpers (data, backtests, agent, pricing, GBM class, charts)
regression_models_comparison.ipynb   01 · regression and classification strategies vs benchmark
time_series.ipynb                    02 · AAPL feature engineering and charts
rl_trading_agent.ipynb               03 · DQN stock-trading agent
gbm_options_rl_agent.ipynb           04 · GBM pricing, volatility forecasts, options DQN agent
risk_parity_backtest.ipynb           +  · inverse-volatility risk parity
AAPL_trading_features.csv            agent input produced by time_series.ipynb
gbm-options-rl-agent-design.md       design notes for the GBM options RL agent
```

*Results come from historical backtests and are shared for research and education only. They are not investment advice.*

---

Built by Damion Marshall with Claude Assistance
