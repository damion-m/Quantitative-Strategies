"""Shared helpers for the quant notebooks.

The code now lives in the ``quantlib`` package, split by the sections this file used to have.
This module re-exports every name it used to define, so ``import func_lib`` followed by
``func_lib.name(...)``, and ``from func_lib import name``, keep working unchanged.

Layout
    quantlib/_common.py        constants shared across modules (TRADING_DAYS)
    quantlib/theme.py          1. Plotly dark theme (applied on import) and small chart helpers
    quantlib/cross_section.py  2. Cross-sectional data and backtest helpers (logistic / lasso notebooks)
    quantlib/timeseries.py     3. Time-series features and charts (time-series notebook)
    quantlib/clustering.py     4. Clustering helpers
    quantlib/rl.py             5. Reinforcement-learning trading agent (RL notebook)
    quantlib/options.py        6. Options pricing: Black-Scholes, Greeks, implied vol, GBM paths,
                                  Monte Carlo and the GBM class (GBM options notebook)
    quantlib/vol.py            6. Volatility: variance proxy, GARCH(1,1), HAR-RV, forecast metrics
    quantlib/portfolio.py      6. Performance statistics, stats table figure, risk parity and
                                  portfolio backtests
    quantlib/options_rl.py     7. RL options trading agent: buy / sell calls and puts with trade
                                  tracking (GBM options notebook)

Importing func_lib still applies the dark Plotly theme, as before.

Heavy optional dependencies (keras, yfinance) are imported lazily, so importing
func_lib only requires numpy, pandas and plotly.
"""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.special import expit   # overflow-safe sigmoid
from scipy.optimize import brentq, minimize
from scipy.signal import lfilter
from scipy.stats import norm

from quantlib._common import (  # noqa: F401
    TRADING_DAYS,
)
from quantlib.theme import (  # noqa: F401
    BG, PALETTE, UP, DOWN, NEUTRAL, RANGE_SELECTOR, THEME_MARGIN, set_plotly_theme, subtitled,
    has_subtitle, _rgba,
)
from quantlib.cross_section import (  # noqa: F401
    createHistPrices, computingReturns, _perf_summary, plot_cumulative_returns,
    plot_calendar_returns, compute_BM_Perf, compute_strat_perf, daily_returns_from_cum,
    _display_name, strategy_comparison_figure, strategy_stats_table,
    coefficient_comparison_figure, chronological_split, scale_features, date_based_cv_splits,
    build_pred_frame, coefficient_bar_figure, calculate_rsi,
)
from quantlib.timeseries import (  # noqa: F401
    download_ohlcv, wilder_rsi, build_ts_features, line_figure, candlestick_volume_figure,
    moving_average_figure, bollinger_figure, indicator_panel_figure, animated_line_figure,
)
from quantlib.clustering import (  # noqa: F401
    find_elbow_point, plot_optimal_cluster_point, train_agent,
)
from quantlib.rl import (  # noqa: F401
    ACTIONS, format_price, progress, build_states, build_dqn, epsilon_decay_for, Agent,
    N_POSITION_FEATURES, position_features, run_episode, print_summary,
    policy_comparison_figure, policy_stats_table, plot_behavior, plot_losses, trade_report,
    trade_statistics, print_trade_statistics, trade_analysis_figure,
)
from quantlib.options import (  # noqa: F401
    download_close, _as_float_arrays, _d1_d2, bs_price, bs_greeks, implied_vol, gbm_paths,
    mc_option_price, gbm_expected_value, GBM,
)
from quantlib.vol import (  # noqa: F401
    daily_variance_proxy, forward_realized_variance, _garch_h_next, fit_garch11, garch_forecast,
    har_design, _refit_blocks, walk_forward_har, walk_forward_garch, forecast_metrics,
)
from quantlib.portfolio import (  # noqa: F401
    performance_stats, stats_table_figure, inverse_vol_weights, backtest_weights,
    weights_area_figure, equity_drawdown_figure,
)
from quantlib.options_rl import (  # noqa: F401
    OPTION_ACTIONS, _OPEN_ACTIONS, POSITION_ORDER, POSITION_STYLE, N_BOOK_FEATURES,
    short_margin_requirement, _book_features, run_options_episode, options_trade_statistics,
    print_options_summary, plot_options_behavior, options_trade_analysis_figure,
)
