"""Section 2 of the original func_lib: cross-sectional data and backtest helpers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import TRADING_DAYS
from .theme import DOWN, NEUTRAL, PALETTE, RANGE_SELECTOR, UP
from .portfolio import performance_stats


# =============================================================================
# 2. Cross-sectional data and backtest helpers
# =============================================================================
def create_hist_prices(start_date='2000-01-01', end_date='2004-05-01', min_obs_per_ticker=100):
    """Adjusted close prices for current S&P 500 members (Date x Ticker).

    Also available under its original name, createHistPrices.
    """
    import yfinance as yf

    headers = {'User-Agent': 'Mozilla/5.0'}
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies',
                         storage_options=headers)[0]['Symbol'].tolist()
    # Drop class-B share tickers (e.g. BRK.B, BF.B): their Wikipedia symbols don't match yfinance
    tickers = [t for t in sp500 if '.B' not in t]

    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    prices = prices.xs('Adj Close', axis=1, level=0)

    # Keep tickers with at least `min_obs_per_ticker` non-missing prices
    return prices.loc[:, prices.count() >= min_obs_per_ticker]


def compute_returns(historical_prices, list_of_momentums, forecast_horizon=1):
    """Long-format (Ticker, Date) frame with the forward return target and momentum features.

    Columns: F_{h}_d_returns (next-h-day return, the target) and {m}_d_returns for each momentum m.
    Also available under its original name, computingReturns.
    """
    fwd = historical_prices.pct_change(forecast_horizon, fill_method=None).shift(-forecast_horizon)
    columns = {f'F_{forecast_horizon}_d_returns': fwd.unstack()}
    for m in list_of_momentums:
        columns[f'{m}_d_returns'] = historical_prices.pct_change(m, fill_method=None).unstack()
    # One concat instead of a chain of outer merges; same rows after dropna
    return pd.concat(columns, axis=1).dropna(how='any')


# Original camelCase names, kept so existing notebooks work unchanged (the same function objects)
createHistPrices = create_hist_prices
computingReturns = compute_returns


def _perf_summary(daily_returns, label):
    """Print CAGR and Sharpe for a daily return series; return (cum_returns, calendar_returns_pct)."""
    cum = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    cagr = (cum.iloc[-1] ** (1 / years) - 1) * 100            # growth of $1 from the start
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)
    print(f' The Model is: {label}')
    print(f' The CAGR is: {cagr:.2f}%')
    print(f' Sharpe Ratio of Strategy: {sharpe:.2f}')
    years_idx = daily_returns.index.get_level_values(0).year
    calendar = (1 + daily_returns).groupby(years_idx).prod().sub(1).mul(100)
    return cum, calendar


def plot_cumulative_returns(cum_returns, title='Cumulative Returns Over Time'):
    fig = go.Figure()
    for i, col in enumerate(cum_returns.columns):
        is_bm = i == 0
        fig.add_scatter(x=cum_returns.index, y=cum_returns[col], name=col, mode='lines',
                        line=dict(width=2.6 if is_bm else 1.8, color=NEUTRAL if is_bm else PALETTE[(i - 1) % len(PALETTE)],
                                  dash='dot' if is_bm else 'solid'),
                        hovertemplate='%{y:.3f}')
    fig.update_layout(title=title, height=520, yaxis_title='Growth of $1',
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    fig.show()


def plot_calendar_returns(calendar_returns, title='Calendar-Year Returns (%)'):
    fig = go.Figure()
    for i, col in enumerate(calendar_returns.columns):
        is_bm = i == 0
        fig.add_bar(x=calendar_returns.index.astype(str), y=calendar_returns[col], name=col,
                    marker_color=NEUTRAL if is_bm else PALETTE[(i - 1) % len(PALETTE)],
                    hovertemplate='%{y:.1f}%')
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(title=title, height=460, barmode='group', yaxis_title='Return (%)',
                      yaxis_ticksuffix='%', hovermode='x unified')
    fig.show()


def compute_BM_Perf(total_returns, plot=True):
    """Equal-weighted benchmark of all stocks: prints CAGR/Sharpe, plots (optional), returns (cum, calendar)."""
    daily_mean = total_returns['F_1_d_returns'].groupby(level='Date').mean().rename('S&P500')
    cum, calendar = _perf_summary(daily_mean, 'S&P500 (equal-weighted benchmark)')
    cum_returns, calendar_returns = cum.to_frame(), calendar.to_frame()
    if plot:
        plot_cumulative_returns(cum_returns)
        plot_calendar_returns(calendar_returns)
    return cum_returns, calendar_returns


def compute_strat_perf(total_returns, cum_returns, calendar_returns, trading_strategy, model_name, plot=True):
    """Apply `trading_strategy` to model predictions, add the strategy to the comparison and plot it.

    Set plot=False when backtesting several models, then compare them all at once with
    `strategy_comparison_figure`.
    """
    ret_col = f'{model_name}_Return'
    total_returns['Position'] = total_returns[model_name].map(trading_strategy)
    total_returns[ret_col] = total_returns['F_1_d_returns'] * total_returns['Position']

    # Equal-weighted average across stocks on each date
    daily_mean = total_returns[ret_col].groupby(level='Date').mean()
    cum, calendar = _perf_summary(daily_mean, model_name)
    cum_returns.loc[:, ret_col] = cum
    calendar_returns.loc[:, ret_col] = calendar

    if plot:
        plot_cumulative_returns(cum_returns)
        plot_calendar_returns(calendar_returns)
    return cum_returns, calendar_returns


def daily_returns_from_cum(cum_returns):
    """Recover daily returns from growth-of-$1 columns (the first day is measured from $1)."""
    daily = cum_returns.pct_change()
    daily.iloc[0] = cum_returns.iloc[0] - 1
    return daily


def _display_name(col):
    return col[:-len('_Return')] if col.endswith('_Return') else col


def strategy_comparison_figure(cum_returns, calendar_returns, title='Strategies vs Benchmark',
                               benchmark_col='S&P500'):
    """Three-panel comparison of every strategy against the benchmark.

    Panels: growth of $1, drawdown, and calendar-year returns. The benchmark is a dotted grey line
    and grey bars; strategies use the theme palette.
    """
    names = list(cum_returns.columns)
    colors = {c: (NEUTRAL if c == benchmark_col else PALETTE[i % len(PALETTE)])
              for i, c in enumerate([c for c in names if c != benchmark_col])}
    colors[benchmark_col] = NEUTRAL
    fig = make_subplots(rows=3, cols=1, row_heights=[0.48, 0.22, 0.30], vertical_spacing=0.08,
                        subplot_titles=('Growth of $1 (out-of-sample)', 'Drawdown', 'Calendar-year return'))
    for col in names:
        is_bm = col == benchmark_col
        label = _display_name(col) + (' (benchmark)' if is_bm else '')
        eq = cum_returns[col]
        dd = eq / eq.cummax() - 1
        line = dict(color=colors[col], width=2.8 if is_bm else 1.8, dash='dot' if is_bm else 'solid')
        fig.add_scatter(x=eq.index, y=eq, name=label, legendgroup=col, line=line,
                        hovertemplate='%{y:.3f}', row=1, col=1)
        fig.add_scatter(x=dd.index, y=dd, name=label, legendgroup=col, showlegend=False,
                        line=dict(color=colors[col], width=1), hovertemplate='%{y:.1%}', row=2, col=1)
        if col in calendar_returns:
            fig.add_bar(x=calendar_returns.index.astype(str), y=calendar_returns[col], name=label,
                        legendgroup=col, showlegend=False, marker_color=colors[col],
                        hovertemplate='%{y:.1f}%', row=3, col=1)
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1), row=3, col=1)
    fig.update_yaxes(title_text='Growth of $1', row=1, col=1)
    fig.update_yaxes(tickformat='.0%', row=2, col=1)
    fig.update_yaxes(ticksuffix='%', row=3, col=1)
    fig.update_layout(title=title, height=1000, barmode='group', hovermode='x unified')
    return fig


def strategy_stats_table(cum_returns, benchmark_col='S&P500'):
    """performance_stats for every column, plus CAGR and correlation relative to the benchmark."""
    daily = daily_returns_from_cum(cum_returns)
    stats = pd.DataFrame({_display_name(c): performance_stats(daily[c]) for c in daily.columns})
    bm = _display_name(benchmark_col)
    stats.loc['Excess CAGR vs Benchmark'] = stats.loc['CAGR'] - stats.loc['CAGR', bm]
    stats.loc['Correlation to Benchmark'] = [daily[c].corr(daily[benchmark_col]) for c in daily.columns]
    return stats


def coefficient_comparison_figure(coefficients, title='Coefficient Comparison'):
    """Grouped bars of coefficients: rows = features, columns = models."""
    fig = go.Figure()
    for i, model in enumerate(coefficients.columns):
        fig.add_bar(x=coefficients.index, y=coefficients[model], name=model,
                    marker_color=PALETTE[i % len(PALETTE)], hovertemplate='%{x}<br>%{y:.3e}<extra></extra>')
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(title=title, height=460, barmode='group', xaxis_title='Feature',
                      yaxis_title='Coefficient (scaled features)', hovermode='closest')
    return fig


def chronological_split(total_returns, features, target, return_col='F_1_d_returns', train_fraction=0.7):
    """Split a (Ticker, Date) panel by DATE so every test date is after every training date.

    Returns a dict with x_train, x_test, y_train, y_test (Series), fwd_returns_test and split_date.
    """
    dates = total_returns.index.get_level_values('Date')
    unique_dates = dates.unique().sort_values()
    split_date = unique_dates[int(train_fraction * len(unique_dates))]
    train, test = total_returns[dates < split_date], total_returns[dates >= split_date]
    return {'x_train': train[features], 'x_test': test[features],
            'y_train': train[target], 'y_test': test[target],
            'fwd_returns_test': test[return_col], 'split_date': split_date}


def scale_features(x_train, x_test):
    """StandardScaler fitted on the training set only -> (x_train_scaled, x_test_scaled, scaler) as DataFrames."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    wrap = lambda x: pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)   # noqa: E731
    return wrap(x_train), wrap(x_test), scaler


def date_based_cv_splits(index, n_splits=5):
    """Expanding-window CV folds over whole dates for a (Ticker, Date) panel, as a list for sklearn `cv=`.

    Validation dates always come after the training dates, and all tickers on a date stay in one fold.
    """
    from sklearn.model_selection import TimeSeriesSplit

    row_dates = index.get_level_values('Date')
    unique = row_dates.unique().sort_values()
    return [(np.flatnonzero(row_dates.isin(unique[tr])), np.flatnonzero(row_dates.isin(unique[va])))
            for tr, va in TimeSeriesSplit(n_splits=n_splits).split(unique)]


def build_pred_frame(predictions, index, fwd_returns, model_name):
    """Model predictions joined with realized forward returns, in the format compute_strat_perf expects."""
    preds = pd.Series(np.asarray(predictions).ravel(), index=index, name=model_name)
    return pd.concat([preds, fwd_returns], axis=1, join='inner')


def coefficient_bar_figure(coefficients, title='Model Coefficients'):
    """Bar chart of model coefficients: green positive, red negative, grey exactly zero (dropped)."""
    fig = go.Figure(go.Bar(
        x=coefficients.index, y=coefficients.values,
        marker_color=[UP if c > 0 else DOWN if c < 0 else NEUTRAL for c in coefficients],
        text=[f'{c:.2e}' for c in coefficients], textposition='outside',
        hovertemplate='%{x}<br>coef = %{y:.3e}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(title=title, height=480, xaxis_title='Feature', yaxis_title='Coefficient value',
                      hovermode='closest', showlegend=False)
    return fig


def calculate_rsi(returns, window=14):
    """Simple-moving-average RSI of a return / price-change series (NaNs dropped).

    How it differs from `wilder_rsi` (quantlib.timeseries):
      * input: a series of returns or price changes, not prices (wilder_rsi takes the close and
        differences it itself);
      * smoothing: a plain `window`-day mean of gains and losses (Cutler's RSI), where wilder_rsi
        uses Wilder's exponential smoothing (alpha = 1 / window), the convention of most charting
        packages;
      * output: NaNs are dropped, so the index is shorter than the input: the warm-up rows and any
        window with no losses (where wilder_rsi gives 100) are left out.
    No notebook uses this function: every published RSI (build_ts_features' 'RSI' column, used by
    the time-series, RL and options notebooks and the saved feature CSVs) comes from wilder_rsi.
    """
    gain = returns.clip(lower=0)
    loss = -returns.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)     # avoid division by zero
    return (100 - 100 / (1 + rs)).dropna()
