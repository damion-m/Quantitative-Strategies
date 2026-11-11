# Versions

Projects are updated as my knowledge improves. Nothing is overwritten: version 1 of every project is
the notebook as first published (git tag `v1`), and each later version is a new, dated notebook that
sits beside it. A version number goes up when the method or the conclusion changes, not for edits
to wording.

| Project | Version | Date | Notebook | What changed |
|---|---|---|---|---|
| Regression strategies | 1 | 2026 | `regression_models_comparison.ipynb` | Five models on momentum features, long or cash, against an equal-weight benchmark |
| | 2 | 2026-11-11 | `regression_models_revisited.ipynb` | Index membership by date, trading costs, sector-relative ranking with a long-short test, 21-day horizon, significance tests |
| Time-series features | 1 | 2026 | `time_series.ipynb` | Seven price features for AAPL, charted and exported |
| | 2 | 2026-11-11 | `time_series_features_revisited.ipynb`, Part 1 | Tests whether the seven features predict returns or volatility, on AAPL and across the S&P 500 |
| | 3 | 2026-11-11 | `time_series_features_revisited.ipynb`, Part 2 | Eight new features (risk, market state, earnings events, relative return) against a pass mark fixed in advance |
| | 4 | 2026-11-11 | `price_range_forecast.ipynb` | Confirms the fear-and-return finding on new periods; a volatility model, an honest price range, and sizing by forecast risk |
| DQN trading agent | 1 | 2026 | `rl_trading_agent.ipynb` | |
| GBM options agent | 1 | 2026 | `gbm_options_rl_agent.ipynb` | GBM pricing, five volatility forecasts, DQN options agent |
| | 2 | 2026-11-11 | `volatility_forecasts_revisited.ipynb` | Formal comparison of the five forecasts, bias tests, combinations, and whether they time the premium |
| Anomaly detection and regime change | 1 | in progress | `regime_detection_research.ipynb` | |
| Risk parity | 1 | 2026 | `risk_parity_backtest.ipynb` | |
| Shared library | 1 | 2026 | `func_lib.py` | |

Helper modules added with the revisions: `cross_section.py`, `vol_eval.py`, `ts_features.py`,
`range_forecast.py`. Rules fixed before each test are kept in `data_cache/*_prereg.json`.
