"""quantlib: the shared research library behind the quant notebooks.

This package holds the code that used to live in the single file ``func_lib.py``, split by that
file's sections. ``func_lib.py`` remains as a thin module that re-exports every name, so existing
notebooks keep using ``import func_lib`` unchanged.

Modules
    _common        constants shared across modules (TRADING_DAYS)
    theme          Plotly dark theme (applied on import) and small chart helpers
    cross_section  cross-sectional data and backtest helpers
    timeseries     time-series features and charts
    clustering     clustering helpers
    rl             DQN trading agent, TradingEnv and EpisodeResult
    options        Black-Scholes, Greeks, implied vol, GBM paths, Monte Carlo, the GBM class, OptionContract
    vol            variance proxy, GARCH(1,1), HAR-RV, forecast metrics, GarchForecaster, HarForecaster
    portfolio      performance statistics, stats table figure, risk parity, portfolio backtests
    options_rl     RL options trading environment (OptionsEnv, OptionsEpisodeResult) and reporting

Importing the package applies the dark Plotly theme, as importing func_lib always has.
"""
from . import _common, theme   # noqa: F401  theme first: it sets the default Plotly template
from . import (  # noqa: F401
    cross_section, timeseries, clustering, rl, options, vol, portfolio, options_rl,
)
