# GBM Options Pricing & RL Long/Short Call/Put Agent: Design Notes
*Notebook: `gbm_options_rl_agent.ipynb`. Helpers are in `quantlib/options.py`, `quantlib/vol.py` and `quantlib/options_rl.py`, re-exported by `quantlib/func_lib.py`. Built 2026-11-06; selling added the same day.*

## Pipeline
1. **Data:** SPY OHLC (yfinance, adjusted). ^VIX is the implied-vol proxy. ^IRX is the risk-free rate.
2. **GBM fit:** MLE μ and σ, plus a normality check with a histogram and a Q-Q plot. The fit shows fat tails and negative skew.
3. **Volatility forecasts** (walk-forward, refitted each January, 21-day target): Historical 21d, EWMA (λ=0.94), GARCH(1,1) MLE, HAR-RV on range-based realized variance (overnight gap + Garman-Klass), and VIX. Scored on QLIKE, RMSE, bias and Mincer-Zarnowitz R².
4. **Pricing engine:** Black-Scholes, all Greeks, and Monte Carlo with antithetic variates. Verified: MC matches BS within the confidence interval, Greeks match finite differences to 1e-6, and put-call parity and implied-vol round trips are exact.
5. **GBM forecasts:** a 63-day fan chart. The drift μ_t is 50% trailing 12-month momentum plus 50% the risk-free rate; σ is the HAR forecast. A calibration test checks 50% and 90% range coverage and the PIT histogram.
6. **GBM edge:** the real-world expected value of the ATM 1-month call or put, E_μ[payoff] discounted at r (closed form e^{(μ−r)T}·BS(rate=μ)), divided by the market price at IV, minus 1.
7. **DQN agent** (`func_lib.Agent`, same set-up as the AAPL RL notebook). Six actions: hold, buy call, buy put, **sell call, sell put**, close oldest.
   - **Pricing:** longs pay the ask, BS(IV+1 vol pt), and close at the bid. Shorts collect the bid and buy back at the ask. Positions are marked at IV and settle at intrinsic value on expiry.
   - **Margin for naked shorts (Reg-T style):** premium + max(20%·S − OTM, 10%·S), × 100. New trades must fit within free equity (equity − margin used). The oldest short is force-closed when equity < margin used ("Margin call").
   - **Risk controls:** at most 6 open contracts, at most 3 naked shorts, and a **panic filter** (Daniel–Moskowitz) that blocks new shorts when the trailing 21-day return < 0 and IV is above its training 80th percentile.
   - **State:** sigmoid diffs and levels of the scaled features, slot shares for long call / long put / short call / short put, average time left, and margin utilization (6 book features).
   - **Reward:** the daily equity change as a percent of capital.
8. **Benchmarks:** a long/short GBM-edge rule (buy when edge > 5%, sell when the market is more than 15% above GBM value, hold to expiry), a random policy, and SPY buy-and-hold.
9. **Tracking:** a per-trade log with Position/Side/Type, entry/exit, strike, IV, forecast vol, GBM edge, signed position Greeks, P&L, return (shorts: % of premium collected) and exit reason (Agent close / Expired / Margin call / End of data), exported as `options_agent_trade_log.csv`. Also statistics by All / Long / Short / Call / Put / each position, a 2×2 dashboard, a monthly returns heatmap, action counts and today's Q-values (with unavailable actions masked).

## Verification
- Account equity change = sum of trade P&L, to the cent, in every training and test run.
- Short P&L = (entry − exit) × 100. Expiries settle at intrinsic value. A short call has delta < 0 and theta > 0.

## Known limitations / next steps
- VIX is used in place of SPY ATM IV, and it usually runs higher because it includes skew, so selling looks somewhat more profitable than it really was. Options are European, with no dividends or early exercise. Real option chain data is needed before any live use.
- The GBM edge is dominated by the drift assumption (`MU_SHRINK`) and by IV versus forecast vol, i.e. the variance risk premium.
- Short premium is negatively skewed. Compare the panic filter on vs off and different short caps.
- Next steps: defined-risk structures (put spreads, iron condors), vega- or margin-based sizing, Heston or rough-vol dynamics, RFSV as a forecaster, skew from real chains, and training on several underlyings.
