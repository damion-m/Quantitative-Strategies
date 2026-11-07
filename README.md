# Quantitative Strategies Projects

> **Repo description (GitHub "About"):** Four connected quant trading projects in Python, from regression signals and time-series features to a DQN stock-trading agent and a GBM options-pricing RL agent that buys and sells calls and puts, plus a risk-parity backtest. Uses a shared helper library and dark Plotly charts.

The projects form a progression. Each one answers a harder question using what the previous one built:

| # | Project | Question | Notebook(s) |
|---|---------|----------|-------------|
| 01 | Regression models | Can simple models predict tomorrow's direction across the S&P 500? | `regression_models_comparison.ipynb`, `logistic_regression.ipynb`, `lasso_regression.ipynb` |
| 02 | Time-series features | What does one stock's own history show? | `time_series.ipynb` |
| 03 | RL trading agent | Can an agent learn when to buy and sell from those features? | `rl_trading_agent.ipynb` |
| 04 | GBM options RL agent | Can it trade options that it prices itself? | `gbm_options_rl_agent.ipynb` |
| + | Risk parity | How should risk be balanced across asset classes? | `risk_parity_backtest.ipynb` |

All notebooks share `func_lib.py`, which provides the data handling, chronological splits, backtests, performance statistics, the DQN agent and the episode runners, option pricing and volatility models, and a common dark Plotly theme.

---

## 01 · Regression models

Five models forecast the S&P 500's next-day move from 1-, 5-, 15- and 20-day momentum, using data from 2000 onward. Each forecast becomes a long/flat strategy, and every strategy is compared with an equal-weighted benchmark.

- **Models:** linear regression (OLS), Lasso (alpha chosen by cross-validation), logistic regression, an unpruned decision tree, and a decision tree tuned by grid search.
- **Method:** one chronological 70/30 split. Features are scaled on training data only, and cross-validation folds are cut by date, so no model sees the future.
- **Results (out of sample, from Sep 2018):** no model beat the benchmark. Hit rates were 50–52% and ROC-AUC was 0.50–0.51. Lasso came closest, growing $1 to about $3.27 against the benchmark's $3.45, mainly because it stayed long 90% of the time.

## 02 · Time-series features

Builds and charts AAPL features for March 2020 to August 2024: 20/50-day moving averages, Bollinger Bands, RSI, price and return volatility, and volume.

- AAPL rose about 240% and closed above its 50-day average on 62% of days.
- RSI was above 70 on 11% of days and below 30 on 1%, typical of a long uptrend.
- Saves `AAPL_trading_features.csv`, which supplies the seven features the RL agent sees.

## 03 · RL trading agent (DQN)

A deep Q-network trades AAPL with three actions: hold, buy one share (up to five) and sell the oldest share.

- **State:** daily changes in the seven features, plus the agent's own position.
- **Training:** reward is daily mark-to-market P&L, with 5 bp costs, action masking, a target network and a paced exploration schedule. It trains on 2020–2023 data and tests on the following 338 unseen days.
- **Results:**

  | Policy | Test P&L | Max drawdown |
  |---|---|---|
  | Final (episode 5) model | $130 (170 trades) | $64 (smallest) |
  | Episode 2 model | $301 | — |
  | Buy-and-hold | $308 | $163 |
  | Random | $241 | $119 |

  The final model had the smallest drawdown but earned less than both benchmarks. The episode 2 model mostly held its position and nearly matched buy-and-hold.
- **Includes:** a reusable backtest function, a full trade log and post-trade analytics.

## 04 · GBM options RL agent

Extends the agent to options on SPY:

1. **Price model:** geometric Brownian motion, fitted by maximum likelihood, with diagnostics showing where fat tails break it.
2. **Volatility forecasts:** historical, EWMA, GARCH(1,1), HAR-RV and VIX, each refitted walk-forward and scored on QLIKE.
3. **Pricing engine:** Monte Carlo with antithetic variates, checked against Black-Scholes, plus the Greeks.
4. **Edge signal:** the GBM expected value of an option compared with its market price.
5. **DQN agent:** six actions (hold, buy call, buy put, sell call, sell put, close), with bid/ask costs, Reg-T-style margin on short options, forced buy-backs and a crash filter.

**Results:**

- Implied volatility exceeded the volatility that followed on 85% of days, by 3.8 points on average.
- Out of sample from late 2021, a $100k account grew 34% under the agent, with 12% volatility and a 21% maximum drawdown. Over the same period SPY returned 80% with a 25% drawdown, the GBM-edge rule lost 2% after a 46% drawdown, and random trading lost 39%.
- Every trade is logged by position type: long or short, call or put.

## + · Risk parity

Inverse-volatility weights across S&P 500 e-mini, gold and 10-year T-note futures (2015–2025), compared with an equal-weight portfolio. The weights use only the previous day's data, and the rebalance frequency and trading costs are configurable.

- Risk parity grew $1 to about $1.61, against about $2.36 for equal weight.
- Both portfolios had a drawdown of about 16% in 2022, when bonds fell alongside stocks.

---

## Takeaways

- Daily direction is close to unpredictable from simple momentum features, so time in the market drove most regression results.
- The environment matters as much as the network. Position-aware states, the right reward and paced exploration were what made the agents trade sensibly.
- Both RL agents took less risk than buy-and-hold and earned less. The best checkpoint was not the final one, so a validation period should be used to choose models.
- Option sellers collect a real variance risk premium, but margin and crash controls decide whether a strategy survives.

## Getting started

```bash
pip install numpy pandas scipy scikit-learn statsmodels plotly yfinance tensorflow keras nbformat
```

1. Keep `func_lib.py` in the same folder as the notebooks, or in the parent folder.
2. Run `time_series.ipynb` before `rl_trading_agent.ipynb`; it writes the agent's input CSV.
3. The other notebooks download their own data with `yfinance` and need internet access.
4. Each notebook's settings are in its **Configuration** cell, and every code cell starts with a short explanation of what it does.

## Repository layout

```
func_lib.py                          shared helpers (data, backtests, agent, pricing, charts)
regression_models_comparison.ipynb   01 · all regression/classification strategies vs benchmark
logistic_regression.ipynb            01 · logistic regression and decision trees
lasso_regression.ipynb               01 · Lasso return forecasts
time_series.ipynb                    02 · AAPL feature engineering and charts
rl_trading_agent.ipynb               03 · DQN stock-trading agent
gbm_options_rl_agent.ipynb           04 · GBM pricing, volatility forecasts, options DQN agent
risk_parity_backtest.ipynb           +  · inverse-volatility risk parity
AAPL_trading_features.csv            agent input produced by time_series.ipynb
```

*Results are from historical backtests and are for research and education only, not investment advice.*
