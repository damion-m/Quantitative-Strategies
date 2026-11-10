"""Shared helpers for the quant notebooks.

Sections
    1. Plotly dark theme
    2. Cross-sectional data and backtest helpers (logistic / lasso notebooks)
    3. Time-series features and charts (time-series notebook)
    4. Clustering helpers
    5. Reinforcement-learning trading agent (RL notebook)
    6. Options pricing, GBM simulation and volatility forecasting (GBM options notebook)
    7. RL options trading agent: buy / sell calls and puts with trade tracking (GBM options notebook)

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

TRADING_DAYS = 252

# =============================================================================
# 1. Plotly dark theme
# =============================================================================
BG = '#0e1117'
PALETTE = ['#4CC9F0', '#F72585', '#43AA8B', '#F9C74F', '#9D4EDD', '#F8961E', '#90BE6D']
UP, DOWN, NEUTRAL = '#43AA8B', '#F94144', '#8d99ae'

RANGE_SELECTOR = dict(buttons=[
    dict(count=1, label='1M', step='month', stepmode='backward'),
    dict(count=6, label='6M', step='month', stepmode='backward'),
    dict(count=1, label='YTD', step='year', stepmode='todate'),
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(step='all', label='All'),
], bgcolor='#1b1f2a', activecolor='#3a3f4b', font=dict(color='#e6e6e6'),
    x=0, xanchor='left', y=1.02, yanchor='bottom')   # top-left, just above the plot; the legend is at the bottom

THEME_MARGIN = dict(l=60, r=30, t=110, b=50)   # t: room for title + subtitle above the range buttons


def set_plotly_theme():
    """Register the 'quant_dark' template and make it the default for all figures."""
    pio.templates['quant_dark'] = go.layout.Template(layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family='Inter, "Segoe UI", Helvetica, Arial, sans-serif', size=13, color='#e6e6e6'),
        # Title pinned to the top edge, so a subtitle line grows downward into the margin rather than
        # into the range buttons / legend / subplot titles that sit just above the plot area
        title=dict(font=dict(size=20, color='#ffffff'), x=0.01, xanchor='left',
                   yref='container', y=1, yanchor='top', pad=dict(t=14),
                   subtitle=dict(font=dict(size=13, color='#9aa3b2'))),
        colorway=PALETTE,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1b1f2a', bordercolor='#3a3f4b', font=dict(color='#e6e6e6')),
        # Legend centred along the bottom edge of the figure. At the top it collides with the range buttons,
        # subplot titles and long titles once the window is narrow; down here Plotly grows the bottom
        # margin to fit it, so it can wrap onto as many rows as it needs at any width.
        legend=dict(orientation='h', yref='container', y=0, yanchor='bottom', xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)'),
        margin=THEME_MARGIN,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', zeroline=False,
                   showline=True, linecolor='#3a3f4b'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', zeroline=False,
                   showline=True, linecolor='#3a3f4b'),
    ))
    pio.templates.default = 'plotly_dark+quant_dark'


set_plotly_theme()   # applied on import so every func_lib chart is themed


def subtitled(title, subtitle):
    """A chart title with a smaller subtitle line underneath, e.g. title=subtitled('GBM edge', 'value / price − 1').

    Plotly can't wrap a long title, so in a narrow window it gets cut off. Keep the headline short
    and put the detail (parameters, dates, how to read the chart) in the subtitle.
    """
    return dict(text=title, subtitle=dict(text=subtitle))


def has_subtitle(title):
    return isinstance(title, dict) and bool(title.get('subtitle', {}).get('text'))


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


# =============================================================================
# 2. Cross-sectional data and backtest helpers
# =============================================================================
def createHistPrices(start_date='2000-01-01', end_date='2004-05-01', min_obs_per_ticker=100):
    """Adjusted close prices for current S&P 500 members (Date x Ticker)."""
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


def computingReturns(historical_prices, list_of_momentums, forecast_horizon=1):
    """Long-format (Ticker, Date) frame with the forward return target and momentum features.

    Columns: F_{h}_d_returns (next-h-day return, the target) and {m}_d_returns for each momentum m.
    """
    fwd = historical_prices.pct_change(forecast_horizon, fill_method=None).shift(-forecast_horizon)
    columns = {f'F_{forecast_horizon}_d_returns': fwd.unstack()}
    for m in list_of_momentums:
        columns[f'{m}_d_returns'] = historical_prices.pct_change(m, fill_method=None).unstack()
    # One concat instead of a chain of outer merges; same rows after dropna
    return pd.concat(columns, axis=1).dropna(how='any')


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

    For Wilder's (exponential) RSI, the one used by most charting packages, see `wilder_rsi`.
    """
    gain = returns.clip(lower=0)
    loss = -returns.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)     # avoid division by zero
    return (100 - 100 / (1 + rs)).dropna()


# =============================================================================
# 3. Time-series features and charts
# =============================================================================
def download_ohlcv(ticker, start, end):
    """Single-ticker OHLCV from yfinance with flat columns and a 'Date' index."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):        # newer yfinance returns (Price, Ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    raw.columns.name = None
    raw.index.name = 'Date'
    return raw.dropna()


def wilder_rsi(close, window=14):
    """Wilder RSI, numerically identical to ta.momentum.RSIIndicator(close, window).rsi()."""
    diff = close.diff()
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)
    ema_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    ema_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    values = np.where(ema_down == 0, 100, 100 - 100 / (1 + ema_up / ema_down))
    return pd.Series(values, index=close.index)


def build_ts_features(df, sma_short=20, sma_long=50, bb_window=20, bb_k=2, rsi_window=14):
    """Add returns, SMAs, volatility, Bollinger Bands, RSI and the next-day direction target.

    Drops the warm-up rows (NaN features) and the final row (no next-day target).
    """
    out = df.copy()
    close = out['Close']

    out['Daily_Return'] = close.pct_change()
    out['Log_Return'] = np.log(close).diff()
    out[f'SMA_{sma_short}'] = close.rolling(sma_short).mean()
    out[f'SMA_{sma_long}'] = close.rolling(sma_long).mean()

    # Rolling std of PRICE in dollars: the quantity Bollinger Bands use (name kept for compatibility)
    out['Volatility'] = close.rolling(bb_window).std()
    # Rolling std of daily RETURNS, annualized: the conventional volatility measure
    out['Return_Volatility'] = out['Daily_Return'].rolling(bb_window).std() * np.sqrt(TRADING_DAYS)

    bb_mid = close.rolling(bb_window).mean()
    out['BB_upper'] = bb_mid + bb_k * out['Volatility']
    out['BB_lower'] = bb_mid - bb_k * out['Volatility']

    out['RSI'] = wilder_rsi(close, rsi_window)
    out['Target'] = (close.shift(-1) > close).astype(int)
    return out.iloc[:-1].dropna()


def line_figure(series, title, yaxis_title='Price (USD)', height=520):
    """Line chart from {label: series or (series, line_kwargs)}."""
    fig = go.Figure()
    for label, s in series.items():
        s, line = s if isinstance(s, tuple) else (s, {})
        fig.add_scatter(x=s.index, y=s.values, name=label, mode='lines', line={'width': 1.8, **line})
    fig.update_layout(title=title, height=height, yaxis_title=yaxis_title,
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


def candlestick_volume_figure(df, title):
    """OHLC candlesticks with a volume panel coloured by up/down day."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'],
                                 close=df['Close'], name='OHLC',
                                 increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    fig.add_bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.6,
                marker_color=np.where(df['Close'] >= df['Open'], UP, DOWN), row=2, col=1)
    fig.update_layout(title=title, height=650, showlegend=False, xaxis_rangeslider_visible=False,
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    return fig


def moving_average_figure(df, title, short_col='SMA_20', long_col='SMA_50'):
    """Close with short/long SMAs; shaded area = spread between them (converge / diverge)."""
    fig = line_figure({'Close': (df['Close'], {'color': '#e6e6e6', 'width': 1.2}),
                       long_col.replace('_', ' '): (df[long_col], {'color': PALETTE[1], 'dash': 'dash'})}, title)
    fig.add_scatter(x=df.index, y=df[short_col], name=short_col.replace('_', ' '), mode='lines',
                    line=dict(color=PALETTE[0], width=1.8), fill='tonexty', fillcolor=_rgba(PALETTE[0], 0.12))
    return fig


def bollinger_figure(df, title):
    """Close price inside a shaded Bollinger Band envelope."""
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df['BB_lower'], name='BB Lower', mode='lines',
                    line=dict(color=PALETTE[4], width=1))
    fig.add_scatter(x=df.index, y=df['BB_upper'], name='BB Upper', mode='lines',
                    line=dict(color=PALETTE[4], width=1), fill='tonexty', fillcolor=_rgba(PALETTE[4], 0.12))
    fig.add_scatter(x=df.index, y=df['Close'], name='Close', mode='lines', line=dict(color='#e6e6e6', width=1.4))
    fig.update_layout(title=title, height=520, yaxis_title='Price (USD)', xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


def indicator_panel_figure(df, title, vol_window=20, rsi_window=14):
    """Stacked panel: annualized return volatility, RSI with 70/30 zones, and volume."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=(f'{vol_window}-Day Return Volatility (annualized)',
                                        f'RSI ({rsi_window})', 'Trading Volume'))
    fig.add_scatter(x=df.index, y=df['Return_Volatility'], name='Volatility', mode='lines',
                    line=dict(color=PALETTE[5]), fill='tozeroy', fillcolor=_rgba(PALETTE[5], 0.12), row=1, col=1)
    fig.add_scatter(x=df.index, y=df['RSI'], name='RSI', mode='lines', line=dict(color=PALETTE[4]), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.10, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=UP, opacity=0.10, line_width=0, row=2, col=1)
    for level, color in [(70, DOWN), (30, UP)]:
        fig.add_hline(y=level, line=dict(dash='dash', color=color, width=1), row=2, col=1)
    fig.add_bar(x=df.index, y=df['Volume'], name='Volume', marker_color=PALETTE[0], opacity=0.6, row=3, col=1)
    fig.update_yaxes(tickformat='.0%', row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(title=title, height=820, showlegend=False)
    return fig


def animated_line_figure(series, title, max_frames=120, frame_ms=50):
    """Line chart that draws itself with Play / Pause buttons.

    Frames are capped at `max_frames`, and axis ranges are fixed so the view doesn't jump.
    """
    step = max(1, len(series) // max_frames)
    frame_ends = list(range(step, len(series), step)) + [len(series)]
    frames = [go.Frame(data=[go.Scatter(x=series.index[:k], y=series.values[:k])]) for k in frame_ends]
    pad = 0.05 * (series.max() - series.min())

    fig = go.Figure(data=[go.Scatter(x=series.index, y=series.values, mode='lines', name=series.name or 'value',
                                     line=dict(color=PALETTE[0], width=2),
                                     fill='tozeroy', fillcolor=_rgba(PALETTE[0], 0.08))],
                    frames=frames)
    fig.update_layout(
        title=title, height=520,
        xaxis=dict(title='Date', range=[series.index[0], series.index[-1]]),
        yaxis=dict(title='Price (USD)', range=[series.min() - pad, series.max() + pad]),
        updatemenus=[dict(type='buttons', showactive=False, x=0, y=1.12, xanchor='left', direction='left',
                          bgcolor='#1b1f2a', font=dict(color='#e6e6e6'), buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, {'frame': {'duration': frame_ms, 'redraw': False}, 'fromcurrent': True}]),
            dict(label='❚❚ Pause', method='animate',
                 args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]),
        ])],
    )
    return fig


# =============================================================================
# 4. Clustering helpers
# =============================================================================
def find_elbow_point(sse, k_range):
    """Elbow = the k whose (normalized) SSE point lies farthest from the first-to-last chord."""
    sse = np.asarray(sse, dtype=float)
    k = np.asarray(k_range, dtype=float)
    sse_normalized = (sse - sse.min()) / (sse.max() - sse.min())
    k_normalized = (k - k.min()) / (k.max() - k.min())

    # Perpendicular distance to the chord, vectorized (2-D cross product written out:
    # np.cross on 2-D vectors is deprecated in NumPy 2)
    p1 = np.array([k_normalized[0], sse_normalized[0]])
    d = np.array([k_normalized[-1], sse_normalized[-1]]) - p1
    distances = np.abs(d[0] * (sse_normalized - p1[1]) - d[1] * (k_normalized - p1[0])) / np.linalg.norm(d)

    elbow_idx = int(np.argmax(distances))
    optimal_k = k_range[elbow_idx]
    print(f'Optimal number of clusters: {optimal_k}')
    return optimal_k, list(distances), k_normalized, sse_normalized


def plot_optimal_cluster_point(sse, k_range):
    optimal_k, distances, k_normalized, sse_normalized = find_elbow_point(sse, k_range)
    elbow_idx = list(k_range).index(optimal_k)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Normalized SSE and Distance to Chord'))
    fig.add_scatter(x=list(k_range), y=list(sse), mode='lines+markers', name='SSE',
                    line=dict(color=PALETTE[0]), row=1, col=1)
    fig.add_scatter(x=k_normalized, y=sse_normalized, mode='lines+markers', name='Normalized SSE',
                    line=dict(color=PALETTE[0]), row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[sse_normalized[0], sse_normalized[-1]], mode='lines', name='Reference line',
                    line=dict(color=PALETTE[1], dash='dash'), row=1, col=2)

    # Perpendicular connectors from each point to the chord
    p1 = np.array([0, sse_normalized[0]])
    v = np.array([1, sse_normalized[-1]]) - p1
    for xk, yk in zip(k_normalized, sse_normalized):
        proj = p1 + np.dot(np.array([xk, yk]) - p1, v) / np.dot(v, v) * v
        fig.add_scatter(x=[xk, proj[0]], y=[yk, proj[1]], mode='lines', showlegend=False, hoverinfo='skip',
                        line=dict(color=NEUTRAL, dash='dot', width=1), row=1, col=2)

    # Highlight the elbow (the original highlighted argmax(SSE), i.e. always the first k)
    for col, x, y in [(1, k_range[elbow_idx], sse[elbow_idx]), (2, k_normalized[elbow_idx], sse_normalized[elbow_idx])]:
        fig.add_scatter(x=[x], y=[y], mode='markers', name=f'Elbow (k = {optimal_k})', showlegend=col == 1,
                        marker=dict(color=DOWN, size=14, line=dict(color='white', width=1.5)), row=1, col=col)
    fig.update_xaxes(title_text='Number of Clusters', row=1, col=1)
    fig.update_yaxes(title_text='Sum of Squared Errors', row=1, col=1)
    fig.update_xaxes(title_text='Normalized Number of Clusters', row=1, col=2)
    fig.update_yaxes(title_text='Normalized SSE', row=1, col=2)
    fig.update_layout(height=460, hovermode='closest')
    fig.show()


def train_agent(env, agent, episodes=1000):
    """Generic tabular Q-learning loop: env.reset()/env.step(); agent.choose_action()/agent.learn()."""
    for episode in range(episodes):
        state = tuple(env.reset())
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            agent.learn(state, action, reward, next_state)
            state = next_state
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1} completed')


# =============================================================================
# 5. Reinforcement-learning (DQN) trading agent
# =============================================================================
ACTIONS = {0: 'hold', 1: 'buy', 2: 'sell'}


def format_price(n):
    return ('-$' if n < 0 else '$') + f'{abs(n):,.2f}'


def progress(iterable, desc='', total=None, min_interval=0.5, width=20):
    """Iterate with a one-line progress readout that updates in place, then ends as a summary line.

    In Jupyter / VS Code it redraws a single display output (display_id), so the saved notebook keeps
    one line instead of one per refresh, and needs no widgets. A text bar such as tqdm relies on
    carriage returns, which only rewrite within one output block, so saved notebooks show hundreds
    of lines. In a terminal it redraws one line with a carriage return.
    """
    import sys
    import time
    total = len(iterable) if total is None else total
    try:
        from IPython import get_ipython
        from IPython.display import display
        in_notebook = getattr(get_ipython(), 'kernel', None) is not None
    except ImportError:
        in_notebook = False

    def status(done, elapsed):
        frac = done / total if total else 1.0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        filled = round(width * frac)
        return (f'{desc}: {frac:4.0%} |{"█" * filled}{" " * (width - filled)}| {done:,}/{total:,} '
                f'[{elapsed:.0f}s elapsed, ~{eta:.0f}s left, {rate:,.0f} it/s]')

    if in_notebook:
        handle = display({'text/plain': status(0, 0.0)}, raw=True, display_id=True)
        def show(text):
            handle.update({'text/plain': text}, raw=True)
    else:
        def show(text):   # pad so a shorter line fully overwrites the previous one
            print('\r' + text.ljust(len(status(total, 1.0)) + 10), end='', file=sys.stderr, flush=True)
        show(status(0, 0.0))

    start = last = time.monotonic()
    done = 0
    for item in iterable:
        yield item
        done += 1
        now = time.monotonic()
        if now - last >= min_interval:
            show(status(done, now - start))
            last = now
    show(f'{desc}: {done:,} steps in {time.monotonic() - start:.0f}s')
    if not in_notebook:
        print(file=sys.stderr)


def build_states(features, window_size=1):
    """Vectorized DQN states for every time step -> float32 array (T, window_size * n_features).

    State at t = sigmoid of the last `window_size` one-day feature changes ending at t
    (0.5 = no change). Rows before the start are padded with the first row.
    """
    features = np.asarray(features, dtype=float)
    padded = np.vstack([np.repeat(features[:1], window_size, axis=0), features])
    diffs = np.diff(padded, axis=0)                                                   # (T + W - 1, F)
    windows = np.lib.stride_tricks.sliding_window_view(diffs, window_size, axis=0)    # (T, F, W)
    return expit(windows.transpose(0, 2, 1)).reshape(len(features), -1).astype(np.float32)


def build_dqn(state_size, action_size=len(ACTIONS), learning_rate=1e-3):
    """Dense(32, relu) -> Dense(8, relu) -> linear Q-value per action; MSE loss, Adam."""
    import keras

    model = keras.Sequential([
        keras.Input(shape=(state_size,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(action_size, activation='linear'),
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def epsilon_decay_for(total_steps, epsilon=1.0, epsilon_min=0.01, explore_fraction=0.8):
    """Per-replay-step decay so epsilon reaches epsilon_min after `explore_fraction` of training.

    A fixed 0.995 per step hits the floor after ~900 steps, well inside the first episode, so the
    agent stops exploring before it has seen most of the data.
    """
    steps = max(int(explore_fraction * total_steps), 1)
    return (epsilon_min / epsilon) ** (1 / steps)


class Agent:
    """DQN agent: epsilon-greedy exploration with optional action masking, random-mini-batch
    experience replay, and an optional target network for stable Bellman targets."""

    def __init__(self, state_size, action_size=len(ACTIONS), model_path=None, test_mode=False,
                 memory_size=1000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=1e-3, target_update_every=None):
        import keras

        self._keras = keras
        self.state_size = state_size
        self.action_size = action_size
        self.test_mode = test_mode
        self.memory = deque(maxlen=memory_size)   # (state, action, reward, next_state, done)
        self.inventory = deque()                  # FIFO open lots: (buy_price, buy_index)
        self.gamma = gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.model = (keras.models.load_model(model_path) if model_path
                      else build_dqn(state_size, action_size, learning_rate))
        # Target network: a lagged copy used for the Bellman target, synced every N replay steps
        self.target_update_every = target_update_every
        self.target_model = None
        if target_update_every and not test_mode:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
        self.train_steps = 0

    def _q(self, model, states):
        # Calling the model directly is much faster than model.predict() for small batches
        return self._keras.ops.convert_to_numpy(model(np.atleast_2d(states), training=False))

    def q_values(self, states):
        return self._q(self.model, states)

    def act(self, state, valid_actions=None):
        """Epsilon-greedy action, restricted to `valid_actions` when given."""
        actions = list(range(self.action_size)) if valid_actions is None else list(valid_actions)
        if not self.test_mode and random.random() <= self.epsilon:
            return random.choice(actions)
        q = self.q_values(state)[0]
        return max(actions, key=lambda a: q[a])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def exp_replay(self, batch_size=32):
        """One gradient step on a random mini-batch using the Bellman target."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        targets = self.q_values(states)                                         # online network
        best_next = self._q(self.target_model or self.model, next_states).max(axis=1)
        targets[np.arange(batch_size), actions] = rewards + self.gamma * best_next * (1 - dones)
        loss = self.model.train_on_batch(states, targets)                        # 1 gradient step

        self.train_steps += 1
        if self.target_model is not None and self.train_steps % self.target_update_every == 0:
            self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return float(np.ravel(loss)[0])


N_POSITION_FEATURES = 3


def position_features(inventory, price, max_inventory):
    """Inventory part of the state: share of max inventory held, holding flag, unrealized return."""
    if not inventory:
        return np.zeros(N_POSITION_FEATURES, dtype=np.float32)
    avg_cost = np.mean([p for p, _ in inventory])
    unrealized = np.clip(10 * (price / avg_cost - 1), -1, 1)      # +/-10% move saturates at +/-1
    return np.array([len(inventory) / max_inventory, 1.0, unrealized], dtype=np.float32)


def run_episode(agent, states, prices, train=False, batch_size=32, train_every=1, desc='', verbose=False,
                policy=None, max_inventory=5, reward_mode='mtm', cost_bps=0.0, close_at_end=True):
    """One pass over the data: buy 1 share per buy signal (up to `max_inventory`), sell lots FIFO.

    The agent's state is the market state plus position_features(), so it knows what it holds.
    Invalid actions are masked: no selling with an empty inventory, no buying when it is full.

    reward_mode:
        'mtm'      : daily mark-to-market P&L of the position after the action, in % of price, net of
                     costs. Dense feedback, so holding a losing position is penalized as it happens.
        'realized' : the original reward, max(trade profit, 0) on sells only.
    policy: optional callable(t, state, valid_actions) -> action, for rule-based benchmarks.
    close_at_end: sell any lots still open on the last day ('End of data'), so every buy shows up
                  in the trade log.

    Returns a dict with buys, sells, trades (DataFrame), losses, actions, the cumulative P&L path in
    $ per share ('pnl'), the inventory path, and open_positions (lots left before the final close).
    """
    inventory = deque()
    if agent is not None:
        agent.inventory = inventory
    buys, sells, trades, losses = [], [], [], []
    n = len(states)
    last_t = n - 1
    actions = np.zeros(n, dtype=int)
    pnl = np.zeros(n)                 # cumulative realized + unrealized P&L, $ per share, net of costs
    held = np.zeros(n, dtype=int)
    realized = 0.0

    def full_state(t):
        return np.concatenate([states[t], position_features(inventory, prices[t], max_inventory)])

    def sell(t, reason):
        nonlocal realized
        buy_price, buy_t = inventory.popleft()
        profit = prices[t] - buy_price
        cost = prices[t] * cost_bps / 1e4
        realized += profit - cost
        sells.append(t)
        trades.append({'buy_t': buy_t, 'sell_t': t, 'Buy Price': buy_price, 'Sell Price': prices[t],
                       'Trade Profit': profit, 'Exit Reason': reason})
        if verbose:
            print(f'Sell: {format_price(prices[t])} | Profit: {format_price(profit)}')
        return profit

    state = full_state(0)
    for t in progress(range(last_t), desc=desc):
        valid = [0] + ([1] if len(inventory) < max_inventory else []) + ([2] if inventory else [])
        action = policy(t, state, valid) if policy is not None else agent.act(state, valid)
        actions[t] = action
        price = prices[t]
        realized_reward = 0.0

        if action == 1:                                   # buy 1 share
            inventory.append((price, t))
            realized -= price * cost_bps / 1e4
            buys.append(t)
            if verbose:
                print(f'Buy:  {format_price(price)}')
        elif action == 2:                                 # sell the oldest lot
            realized_reward = max(sell(t, 'Agent sell'), 0.0)

        # Mark the book to t+1
        unrealized_next = sum(prices[t + 1] - p for p, _ in inventory)
        pnl[t + 1] = realized + unrealized_next
        held[t + 1] = len(inventory)
        if reward_mode == 'mtm':
            reward = 100 * (pnl[t + 1] - pnl[t]) / price
        else:
            reward = realized_reward

        next_state = full_state(t + 1)
        if train and agent is not None:
            agent.remember(state, action, reward, next_state, t == last_t - 1)
            if len(agent.memory) > batch_size and t % train_every == 0:
                losses.append(agent.exp_replay(batch_size))
        state = next_state

    open_positions = len(inventory)
    if close_at_end:
        while inventory:
            sell(last_t, 'End of data')
        pnl[last_t] = realized
        held[last_t] = 0

    trades = pd.DataFrame(trades, columns=['buy_t', 'sell_t', 'Buy Price', 'Sell Price', 'Trade Profit',
                                           'Exit Reason'])
    trades = trades.astype({'buy_t': int, 'sell_t': int, 'Trade Profit': float})   # also when empty
    return {'buys': buys, 'sells': sells, 'trades': trades, 'losses': losses, 'actions': actions,
            'pnl': pnl, 'inventory': held, 'open_positions': open_positions}


def print_summary(result, title):
    profits = result['trades']['Trade Profit']
    counts = np.bincount(result['actions'][:-1], minlength=3) if 'actions' in result else None
    print('-' * 60)
    print(title)
    print(f'Closed trades: {len(profits):,} | open at end (closed on last day): {result["open_positions"]:,}')
    print(f'Total Profit:  {format_price(profits.sum())}  (net of costs: '
          f'{format_price(result["pnl"][-1]) if "pnl" in result else "n/a"})')
    print(f'Total Winners: {format_price(profits[profits > 0].sum())}')
    print(f'Total Losers:  {format_price(profits[profits <= 0].sum())}')
    if counts is not None:
        print(f'Actions: hold {counts[0]:,} | buy {counts[1]:,} | sell {counts[2]:,}')
    if result['losses']:
        print(f'Max batch loss: {max(result["losses"]):.4f}')
    print('-' * 60)


def policy_comparison_figure(dates, results, prices=None, title='Cumulative P&L ($ per share)'):
    """Cumulative P&L of several run_episode results (plus price) on one chart, with inventory held."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=('Cumulative P&L, net of costs ($ per share)', 'Shares held'))
    for i, (name, res) in enumerate(results.items()):
        color = NEUTRAL if 'hold' in name.lower() and 'buy' in name.lower() else PALETTE[i % len(PALETTE)]
        dash = 'dot' if color == NEUTRAL else 'solid'
        fig.add_scatter(x=dates, y=res['pnl'], name=name, legendgroup=name,
                        line=dict(color=color, width=2.2, dash=dash), row=1, col=1)
        fig.add_scatter(x=dates, y=res['inventory'], name=name, legendgroup=name, showlegend=False,
                        line=dict(color=color, width=1, shape='hv'), row=2, col=1)
    fig.update_yaxes(tickprefix='$', row=1, col=1)
    fig.update_layout(title=title, height=640, xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


def policy_stats_table(results):
    """One row of headline stats per policy from run_episode results."""
    rows = {}
    for name, res in results.items():
        profits = res['trades']['Trade Profit']
        pnl = pd.Series(res['pnl'])
        dd = (pnl.cummax().clip(lower=0) - pnl).max()
        n_win, n_loss = int((profits > 0).sum()), int((profits <= 0).sum())
        counts = np.bincount(res['actions'][:-1], minlength=3)
        rows[name] = {'Net P&L ($/share)': pnl.iloc[-1], 'Closed Trades': len(profits),
                      'Win Rate': n_win / len(profits) if len(profits) else np.nan,
                      'Avg Trade ($)': profits.mean() if len(profits) else np.nan,
                      'Max Drawdown ($/share)': dd,
                      'Profit Factor': (profits[profits > 0].sum() / -profits[profits <= 0].sum()
                                        if n_loss and profits[profits <= 0].sum() < 0 else np.nan),
                      'Days Holding': float((np.asarray(res['inventory']) > 0).mean()),
                      'Buy Actions': int(counts[1]), 'Sell Actions': int(counts[2])}
    return pd.DataFrame(rows)


def plot_behavior(df, result, title):
    """Price, Bollinger Bands and buy/sell markers for one episode."""
    close = df['Close']
    profit = result['trades']['Trade Profit'].sum()
    fig = go.Figure()
    if {'BB_upper', 'BB_lower'} <= set(df.columns):
        fig.add_scatter(x=df.index, y=df['BB_lower'], name='Bollinger Bands', mode='lines',
                        line=dict(color=PALETTE[4], width=1), legendgroup='bb')
        fig.add_scatter(x=df.index, y=df['BB_upper'], mode='lines', showlegend=False, legendgroup='bb',
                        line=dict(color=PALETTE[4], width=1), fill='tonexty',
                        fillcolor=_rgba(PALETTE[4], 0.10), hoverinfo='skip')
    fig.add_scatter(x=df.index, y=close, name='Close Price', mode='lines', line=dict(color='#e6e6e6', width=1.5))
    for idx, name, symbol, color in [(result['buys'], 'Buy', 'triangle-up', UP),
                                     (result['sells'], 'Sell', 'triangle-down', DOWN)]:
        fig.add_scatter(x=df.index[idx], y=close.iloc[idx], name=f'{name} signal', mode='markers',
                        marker=dict(symbol=symbol, size=10, color=color, line=dict(width=0.5, color='black')))
    fig.update_layout(title=subtitled(title, f'Total gains: {format_price(profit)}'), height=520,
                      yaxis_title='Price (USD)', xaxis=dict(rangeselector=RANGE_SELECTOR))
    fig.show()


def plot_losses(losses, title, smooth_window=50):
    fig = go.Figure(go.Scatter(y=losses, mode='lines', line=dict(color=PALETTE[0], width=1), name='MSE loss'))
    if len(losses) >= smooth_window:
        fig.add_scatter(y=pd.Series(losses).rolling(smooth_window).mean(), mode='lines',
                        line=dict(color=PALETTE[3], width=2.5), name=f'{smooth_window}-batch mean')
    fig.update_layout(title=title, height=400, xaxis_title='Batch', yaxis_title='MSE loss')
    fig.show()


def trade_report(df, result):
    """Closed trades with timestamps and the market context at the time of the buy."""
    trades = result['trades'].copy()
    trades['Buy Timestamp'] = df.index[trades['buy_t']]
    trades['Sell Timestamp'] = df.index[trades['sell_t']]
    context = {'SMA_20': 'Buy SMA20', 'SMA_50': 'Buy SMA50', 'Volatility': 'Buy Volatility'}
    for col, label in context.items():
        if col in df.columns:
            trades[label] = df[col].to_numpy()[trades['buy_t']]
    trades['Winner'] = trades['Trade Profit'] > 0
    trades['Holding Days'] = trades['sell_t'] - trades['buy_t']
    return trades.drop(columns=['buy_t', 'sell_t'])


def trade_statistics(trade_tracker):
    """Worst trade, max drawdown of cumulative realized P&L, win/loss ratio and profit factor."""
    profits = trade_tracker['Trade Profit']
    realized = (trade_tracker.sort_values('Sell Timestamp')
                .set_index('Sell Timestamp')['Trade Profit'].cumsum())
    n_win, n_loss = int((profits > 0).sum()), int((profits <= 0).sum())
    gross_win, gross_loss = profits[profits > 0].sum(), -profits[profits <= 0].sum()
    return {
        'closed_trades': len(profits),
        'worst_trade': profits.min() if len(profits) else 0.0,
        'max_drawdown': (realized.cummax().clip(lower=0) - realized).max() if len(realized) else 0.0,
        'win_loss_ratio': n_win / n_loss if n_loss else np.nan,
        'profit_factor': gross_win / gross_loss if gross_loss else np.nan,
        'realized_pnl': realized,
    }


def print_trade_statistics(stats):
    print(f'Closed trades:       {stats["closed_trades"]:,}')
    print(f'Worst single trade:  {format_price(stats["worst_trade"])}')
    print(f'Max drawdown (realized P&L): {format_price(stats["max_drawdown"])}')
    for key, label in [('win_loss_ratio', 'Win/Loss ratio:     '), ('profit_factor', 'Profit factor:      ')]:
        value = stats[key]
        print(f'{label} {value:.2f}' if np.isfinite(value) else f'{label} n/a (no losing trades)')


def trade_analysis_figure(trade_tracker, realized_pnl, title='Backtest Trade Analysis'):
    """Cumulative realized P&L plus a winners-vs-losers histogram of volatility at entry."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.08,
                        subplot_titles=('Cumulative Realized P&L', 'Buy Volatility: Winners vs Losers'))
    fig.add_scatter(x=realized_pnl.index, y=realized_pnl.values, mode='lines', name='Realized P&L',
                    line=dict(color=PALETTE[0], width=2), fill='tozeroy', fillcolor=_rgba(PALETTE[0], 0.10),
                    row=1, col=1)
    if 'Buy Volatility' in trade_tracker:
        for winner, name, color in [(True, 'Winner', UP), (False, 'Loser', DOWN)]:
            fig.add_histogram(x=trade_tracker.loc[trade_tracker['Winner'] == winner, 'Buy Volatility'],
                              name=name, marker_color=color, opacity=0.6, nbinsx=30, row=1, col=2)
    fig.update_layout(title=title, height=460, barmode='overlay', hovermode='closest')
    fig.update_yaxes(title_text='$ per share', row=1, col=1)
    fig.update_xaxes(title_text='Volatility at buy', row=1, col=2)
    fig.update_yaxes(title_text='Trades', row=1, col=2)
    return fig


# =============================================================================
# 6. Options pricing, GBM simulation and volatility forecasting
# =============================================================================
from scipy.optimize import brentq, minimize   # noqa: E402
from scipy.signal import lfilter               # noqa: E402
from scipy.stats import norm                   # noqa: E402


def download_close(tickers, start, end=None):
    """Adjusted close prices as a DataFrame with one column per ticker."""
    import yfinance as yf

    close = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    return close.to_frame() if isinstance(close, pd.Series) else close


def _as_float_arrays(*xs):
    return (np.asarray(x, dtype=float) for x in xs)


def _d1_d2(S, K, T, r, sigma, q):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bs_price(S, K, T, r, sigma, q=0.0, kind='call'):
    """Black-Scholes-Merton price of a European 'call', 'put' or 'straddle' (vectorized).

    T in years, r / q / sigma annualized decimals. At T -> 0 the price tends to intrinsic value.
    """
    S, K, T, r, sigma, q = _as_float_arrays(S, K, T, r, sigma, q)
    T, sigma = np.maximum(T, 1e-10), np.maximum(sigma, 1e-8)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    prices = {'call': call, 'put': put, 'straddle': call + put}
    if kind not in prices:
        raise ValueError(f"kind must be 'call', 'put' or 'straddle', got {kind!r}")
    return prices[kind]


def bs_greeks(S, K, T, r, sigma, q=0.0, kind='call'):
    """Black-Scholes Greeks (vectorized).

    Units: delta per $1 of spot, gamma per $1, vega per 1 vol point (0.01),
    theta per calendar day, rho per 1% rate move.
    """
    S, K, T, r, sigma, q = _as_float_arrays(S, K, T, r, sigma, q)
    T, sigma = np.maximum(T, 1e-10), np.maximum(sigma, 1e-8)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r, pdf = np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)

    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf * np.sqrt(T) / 100
    decay = -S * disc_q * pdf * sigma / (2 * np.sqrt(T))
    call = dict(delta=disc_q * norm.cdf(d1), gamma=gamma, vega=vega,
                theta=(decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)) / 365,
                rho=K * T * disc_r * norm.cdf(d2) / 100)
    put = dict(delta=disc_q * (norm.cdf(d1) - 1), gamma=gamma, vega=vega,
               theta=(decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)) / 365,
               rho=-K * T * disc_r * norm.cdf(-d2) / 100)
    if kind == 'call':
        return call
    if kind == 'put':
        return put
    if kind == 'straddle':
        return {g: call[g] + put[g] for g in call}
    raise ValueError(f"kind must be 'call', 'put' or 'straddle', got {kind!r}")


def implied_vol(price, S, K, T, r, q=0.0, kind='call', lo=1e-4, hi=5.0):
    """Black-Scholes implied volatility by Brent's method (scalar). NaN if no solution in [lo, hi]."""
    f = lambda s: float(bs_price(S, K, T, r, s, q, kind)) - price   # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-8)


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None, antithetic=True):
    """Simulate GBM paths dS = mu S dt + sigma S dW exactly in log space -> array (n_steps + 1, n_paths)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        z = rng.standard_normal((n_steps, (n_paths + 1) // 2))
        z = np.concatenate([z, -z], axis=1)[:, :n_paths]
    else:
        z = rng.standard_normal((n_steps, n_paths))
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
    return S0 * np.exp(log_paths)


def mc_option_price(S0, K, T, r, sigma, q=0.0, kind='call', n_paths=100_000, seed=None,
                    antithetic=True, drift=None):
    """Monte Carlo value of a European option under GBM -> (price, standard error).

    drift=None prices under the risk-neutral measure (drift r - q). Passing a real-world
    drift mu returns the expected payoff under that GBM, discounted at r: a forecast value.
    """
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    mu = (r - q) if drift is None else drift
    terminal = lambda zz: S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * zz)   # noqa: E731
    payoff_fn = {'call': lambda s: np.maximum(s - K, 0), 'put': lambda s: np.maximum(K - s, 0),
                 'straddle': lambda s: np.abs(s - K)}[kind]
    payoff = payoff_fn(terminal(z))
    if antithetic:                                   # average antithetic pairs before the std error
        payoff = 0.5 * (payoff + payoff_fn(terminal(-z)))
    disc = np.exp(-r * T)
    return disc * payoff.mean(), disc * payoff.std(ddof=1) / np.sqrt(len(payoff))


def gbm_expected_value(S, K, T, r, mu, sigma, kind='call'):
    """Closed-form real-world GBM forecast value: E_mu[payoff] discounted at r.

    Uses the identity E_mu[payoff] = exp(mu T) * BS(S, K, T, rate=mu, sigma).
    """
    T = np.maximum(np.asarray(T, dtype=float), 1e-10)
    return np.exp((np.asarray(mu) - np.asarray(r)) * T) * bs_price(S, K, T, mu, sigma, 0.0, kind)


@dataclass
class GBM:
    """A geometric Brownian motion dS = mu S dt + sigma S dW with fixed annual mu and sigma.

    A thin, readable wrapper around the tested functions above (gbm_paths, gbm_expected_value),
    for when one set of parameters is in play: fitting a sample, a forecast range, a fan chart.
    Walk-forward code that re-estimates mu and sigma every day can keep calling the functions
    directly, since they accept arrays of parameters.
    """
    mu: float = np.nan      # annual drift (decimal)
    sigma: float = np.nan   # annual volatility (decimal)

    @classmethod
    def fit(cls, log_returns, periods_per_year=TRADING_DAYS):
        """Maximum-likelihood fit from per-period log returns (NaNs ignored).

        sigma = std(log r) * sqrt(periods); mu = mean(log r) * periods + sigma^2 / 2.
        """
        x = np.asarray(log_returns, dtype=float)
        sigma = np.nanstd(x, ddof=1) * np.sqrt(periods_per_year)
        return cls(mu=np.nanmean(x) * periods_per_year + 0.5 * sigma ** 2, sigma=sigma)

    @classmethod
    def fit_prices(cls, prices, periods_per_year=TRADING_DAYS):
        """Maximum-likelihood fit from a price series."""
        return cls.fit(np.diff(np.log(np.asarray(prices, dtype=float))), periods_per_year)

    def forecast(self, S0, T, confidence=0.90):
        """Expected price S0 e^(mu T) and the central `confidence` range of the lognormal S_T."""
        m, s = (self.mu - 0.5 * self.sigma ** 2) * T, self.sigma * np.sqrt(T)
        lo, hi = S0 * np.exp(norm.ppf([(1 - confidence) / 2, (1 + confidence) / 2], loc=m, scale=s))
        return {'expected': S0 * np.exp(self.mu * T), 'median': S0 * np.exp(m),
                'interval': (lo, hi), 'confidence': confidence}

    def simulate(self, S0, T, n_steps, n_paths, seed=None, antithetic=True):
        """Simulated price paths, shape (n_steps + 1, n_paths). See gbm_paths."""
        return gbm_paths(S0, self.mu, self.sigma, T, n_steps, n_paths, seed=seed, antithetic=antithetic)

    def option_value(self, S, K, T, r, kind='call'):
        """Real-world forecast value of a European option, discounted at r. See gbm_expected_value."""
        return gbm_expected_value(S, K, T, r, self.mu, self.sigma, kind)


def daily_variance_proxy(ohlc):
    """Range-based daily variance: overnight gap^2 + Garman-Klass intraday estimator."""
    overnight = np.log(ohlc['Open'] / ohlc['Close'].shift(1)) ** 2
    hl = np.log(ohlc['High'] / ohlc['Low']) ** 2
    co = np.log(ohlc['Close'] / ohlc['Open']) ** 2
    return (overnight + 0.5 * hl - (2 * np.log(2) - 1) * co).rename('rv')


def forward_realized_variance(returns, horizon=21):
    """Mean squared daily return over t+1..t+horizon, aligned at t (the forecast target)."""
    return (returns ** 2).rolling(horizon).mean().shift(-horizon)


# ---- GARCH(1,1) ----
def _garch_h_next(r, omega, alpha, beta, h0):
    """h_next[t] = omega + alpha r_t^2 + beta h_next[t-1] (variance of r_{t+1} given info at t)."""
    x = omega + alpha * r ** 2
    return lfilter([1.0], [1.0, -beta], x, zi=[beta * h0])[0]


def fit_garch11(returns):
    """Gaussian MLE of a GARCH(1,1) on daily returns -> dict(mu, omega, alpha, beta, h0)."""
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    x = 100 * (r - mu)                         # percent returns for numerical stability
    h0 = x.var()

    def neg_loglik(p):
        omega, alpha, beta = p
        h = np.concatenate([[h0], _garch_h_next(x, omega, alpha, beta, h0)[:-1]])
        h = np.maximum(h, 1e-12)
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    res = minimize(neg_loglik, x0=[0.05 * h0, 0.08, 0.90], method='SLSQP',
                   bounds=[(1e-8, None), (0.0, 1.0), (0.0, 1.0)],
                   constraints=[{'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2]}])
    omega, alpha, beta = res.x
    return {'mu': mu, 'omega': omega / 1e4, 'alpha': alpha, 'beta': beta, 'h0': h0 / 1e4}


def garch_forecast(returns, params, horizon=21):
    """Average daily variance over t+1..t+horizon implied by fitted GARCH(1,1), for every t."""
    r = np.asarray(returns, dtype=float) - params['mu']
    h1 = _garch_h_next(r, params['omega'], params['alpha'], params['beta'], params['h0'])
    phi = params['alpha'] + params['beta']
    long_run = params['omega'] / (1 - phi)
    avg = long_run + (h1 - long_run) * (1 - phi ** horizon) / ((1 - phi) * horizon)
    return pd.Series(avg, index=getattr(returns, 'index', None))


# ---- HAR-RV (Corsi 2009) ----
def har_design(rv):
    """HAR regressors: daily, weekly (5d) and monthly (22d) average realized variance."""
    return pd.DataFrame({'rv_d': rv, 'rv_w': rv.rolling(5).mean(), 'rv_m': rv.rolling(22).mean()})


def _refit_blocks(index, first_test_date, refit_freq):
    starts = [index.searchsorted(pd.Timestamp(first_test_date))]
    starts += [index.searchsorted(d) for d in pd.date_range(first_test_date, index[-1], freq=refit_freq)]
    starts = sorted(set(s for s in starts if s < len(index)))
    return list(zip(starts, starts[1:] + [len(index)]))


def walk_forward_har(rv, target, first_test_date, horizon=21, refit_freq='YS'):
    """Out-of-sample HAR forecasts, refitted at each `refit_freq` boundary on data known by then.

    A training row t is used only if its target window (t+1..t+horizon) ended before the refit date.
    """
    X = har_design(rv)
    forecasts = pd.Series(np.nan, index=rv.index, name='HAR')
    for start, end in _refit_blocks(rv.index, first_test_date, refit_freq):
        train = slice(0, max(start - horizon, 0))
        Xt, yt = X.iloc[train], target.iloc[train]
        ok = Xt.notna().all(axis=1) & yt.notna()
        A = np.column_stack([np.ones(ok.sum()), Xt[ok].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, yt[ok].to_numpy(), rcond=None)
        Xp = X.iloc[start:end]
        forecasts.iloc[start:end] = coef[0] + Xp.to_numpy() @ coef[1:]
    floor = np.nanpercentile(rv, 1)                  # keep variance forecasts positive
    return forecasts.clip(lower=floor)


def walk_forward_garch(returns, first_test_date, horizon=21, refit_freq='YS'):
    """Out-of-sample GARCH(1,1) forecasts, refitted on all returns before each refit date."""
    forecasts = pd.Series(np.nan, index=returns.index, name='GARCH')
    clean = returns.fillna(0.0)
    for start, end in _refit_blocks(returns.index, first_test_date, refit_freq):
        params = fit_garch11(clean.iloc[:start])
        forecasts.iloc[start:end] = garch_forecast(clean, params, horizon).iloc[start:end].to_numpy()
    return forecasts


def forecast_metrics(forecasts, target):
    """Compare daily-variance forecasts against realized variance.

    RMSE / MAE / bias are in annualized vol points; QLIKE is the robust loss of Patton (2011),
    lower is better. MZ R^2 is the Mincer-Zarnowitz regression fit.
    """
    rows = {}
    for name, f in forecasts.items():
        ok = f.notna() & target.notna() & (f > 0) & (target > 0)
        fv, yv = f[ok].to_numpy(), target[ok].to_numpy()
        vol_f, vol_y = np.sqrt(fv * TRADING_DAYS) * 100, np.sqrt(yv * TRADING_DAYS) * 100
        ratio = yv / fv
        mz = np.corrcoef(fv, yv)[0, 1] ** 2
        rows[name] = {'RMSE (vol pts)': np.sqrt(np.mean((vol_f - vol_y) ** 2)),
                      'MAE (vol pts)': np.mean(np.abs(vol_f - vol_y)),
                      'Bias (vol pts)': np.mean(vol_f - vol_y),
                      'QLIKE': np.mean(ratio - np.log(ratio) - 1),
                      'MZ R²': mz, 'Obs': int(ok.sum())}
    return pd.DataFrame(rows).T


def performance_stats(returns, rf=0.0, periods=TRADING_DAYS):
    """Annualized performance statistics for a periodic return series."""
    r = pd.Series(returns).dropna()
    rf = pd.Series(rf, index=r.index) if np.isscalar(rf) else pd.Series(rf).reindex(r.index).fillna(0)
    excess = r - rf
    equity = (1 + r).cumprod()
    years = len(r) / periods
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(periods)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(periods)
    max_dd = -(equity / equity.cummax() - 1).min()
    return pd.Series({
        'Total Return': equity.iloc[-1] - 1, 'CAGR': cagr, 'Annualized Return (arith.)': r.mean() * periods,
        'Annualized Volatility': vol, 'Downside Deviation': downside,
        'Sharpe Ratio': excess.mean() * periods / vol if vol else np.nan,
        'Sortino Ratio': excess.mean() * periods / downside if downside else np.nan,
        'Max Drawdown': max_dd, 'Calmar Ratio': cagr / max_dd if max_dd else np.nan,
        'Skewness': r.skew(), 'Positive Days': (r > 0).mean(),
    })


def stats_table_figure(df, title, pct_rows=(), decimals=2, index_name='Metric', height=None):
    """Dark Plotly table of a DataFrame; rows listed in `pct_rows` are shown as percentages."""
    def fmt(row, v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return '–'
        if row in pct_rows:
            return f'{v:.{decimals}%}'
        if float(v).is_integer() and abs(v) >= 1:      # counts (trades, observations, windows)
            return f'{v:,.0f}'
        return f'{v:,.{decimals}f}'

    columns = [[str(i) for i in df.index]] + [[fmt(i, v) for i, v in df[c].items()] for c in df.columns]
    stripes = [[('#11151d', '#161b24')[i % 2] for i in range(len(df))]] * (len(df.columns) + 1)
    fig = go.Figure(go.Table(
        header=dict(values=[index_name, *map(str, df.columns)], fill_color='#1b1f2a', line_color='#3a3f4b',
                    font=dict(color='white', size=13), align='left', height=32),
        cells=dict(values=columns, fill_color=stripes, line_color='#3a3f4b',
                   font=dict(color='#e6e6e6', size=12), align='left', height=28),
    ))
    subtitle_px = 22 if has_subtitle(title) else 0      # room for the subtitle line
    fig.update_layout(title=title, height=height or 110 + subtitle_px + 30 * len(df),
                      margin=dict(t=60 + subtitle_px, b=10, l=20, r=20))
    return fig


# ---- Risk parity / portfolio backtests ----
def inverse_vol_weights(returns, window=36, rebalance=None):
    """Naive risk-parity weights proportional to 1 / rolling volatility (rows sum to 1).

    rebalance: None = daily, or a pandas period alias such as 'W' or 'M'. Weights computed on the
    last trading day of each period are held until the next rebalance.
    """
    inv_vol = 1 / returns.rolling(window).std()
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    if rebalance:
        idx = weights.index
        last_day = idx.to_series().groupby(idx.to_period(rebalance)).transform('max')
        weights = weights.copy()
        weights.loc[np.asarray(idx != last_day)] = np.nan     # keep period-end weights only ...
        weights = weights.ffill()                              # ... and hold them until the next rebalance
    return weights


def backtest_weights(target_weights, returns, cost_bps=0.0):
    """Daily portfolio returns from target weights, lagged one day (no look-ahead), net of costs.

    Returns (portfolio_returns, applied_weights, turnover). Costs = one-way turnover * cost_bps.
    """
    weights = target_weights.shift(1).dropna()
    asset_returns = returns.loc[weights.index]
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    portfolio = (weights * asset_returns).sum(axis=1) - turnover * cost_bps / 1e4
    return portfolio, weights, turnover


def weights_area_figure(weights, title='Portfolio Weights'):
    """Stacked-area chart of portfolio weights over time."""
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_scatter(x=weights.index, y=weights[col], name=col, mode='lines', stackgroup='weights',
                        line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]), hovertemplate='%{y:.1%}')
    fig.update_layout(title=title, height=420, yaxis=dict(title='Weight', tickformat='.0%', range=[0, 1]))
    return fig


def equity_drawdown_figure(returns_by_name, title='Cumulative Returns and Drawdowns'):
    """Growth of $1 (top) and underwater drawdown (bottom) for a dict of daily return series."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=('Growth of $1', 'Drawdown'))
    for i, (name, r) in enumerate(returns_by_name.items()):
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        color = PALETTE[i % len(PALETTE)]
        fig.add_scatter(x=equity.index, y=equity, name=name, legendgroup=name,
                        line=dict(color=color, width=2.2), hovertemplate='%{y:.3f}', row=1, col=1)
        fig.add_scatter(x=drawdown.index, y=drawdown, name=name, legendgroup=name, showlegend=False,
                        fill='tozeroy', line=dict(color=color, width=1), opacity=0.5,
                        hovertemplate='%{y:.1%}', row=2, col=1)
    fig.update_yaxes(tickformat='.0%', row=2, col=1)
    fig.update_layout(title=title, height=680, xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


# =============================================================================
# 7. RL options trading agent: buy / sell calls and puts, close, hold
# =============================================================================
OPTION_ACTIONS = {0: 'hold', 1: 'buy call', 2: 'buy put', 3: 'sell call', 4: 'sell put', 5: 'close oldest'}
_OPEN_ACTIONS = {1: ('call', 1), 2: ('put', 1), 3: ('call', -1), 4: ('put', -1)}   # action -> (kind, sign)
POSITION_ORDER = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
POSITION_STYLE = {   # colour and marker for each position type in the charts
    'Long Call': (UP, 'triangle-up'), 'Long Put': (DOWN, 'triangle-down'),
    'Short Call': (PALETTE[5], 'triangle-down-open'), 'Short Put': (PALETTE[0], 'triangle-up-open'),
}
N_BOOK_FEATURES = 6


def short_margin_requirement(kind, S, K, option_value, mult=100, pct=0.20, min_pct=0.10):
    """Reg-T style margin for one naked short option:
    premium + max(pct * S - out-of-the-money amount, min_pct * S), times the contract multiplier.
    """
    otm = max(K - S, 0.0) if kind == 'call' else max(S - K, 0.0)
    return mult * (option_value + max(pct * S - otm, min_pct * S))


def _book_features(book, t, max_positions, tenor_days, margin_used, equity):
    """Book part of the state: slot share per position type, average time left, margin utilization."""
    counts = [sum(p['Position'] == name for p in book) / max_positions for name in POSITION_ORDER]
    time_left = np.mean([(p['expiry_t'] - t) / tenor_days for p in book]) if book else 0.0
    utilization = min(margin_used / equity, 1.5) if equity > 0 else 1.5
    return np.array(counts + [time_left, utilization], dtype=np.float32)


def run_options_episode(market_states, md, cfg, agent=None, policy=None, train=False,
                        batch_size=32, train_every=1, desc='', verbose=False):
    """Simulate one pass of an options book driven by a DQN `agent` or a `policy(t, state)` callable.

    market_states : (T, n) array of market features
    md            : dict of aligned arrays: dates, S, iv, rf (risk-free rate), sigma_f, mu, call_edge, put_edge, and
                    optionally short_allowed (bool), which blocks NEW short positions where False
    cfg           : tenor_days, moneyness, cost_vol, contract_mult, capital, max_positions, and optionally
                    max_short_positions, short_margin_pct (0.20), min_margin_pct (0.10)

    Actions (OPTION_ACTIONS): 0 hold, 1 buy call, 2 buy put, 3 sell call, 4 sell put, 5 close oldest.
    Each trade is 1 contract of a European option struck at S*(1 +/- moneyness), expiring in tenor_days.
      * Longs pay the ask BS(iv + cost) and close at the bid BS(iv - cost).
      * Shorts collect the bid BS(iv - cost) and buy back at the ask BS(iv + cost). They must post
        Reg-T style margin and are force-closed ('Margin call') if equity falls below margin used.
      * Positions are marked at BS(iv) and settle at intrinsic value on expiry.
    An action that can't be done (book full, not enough free equity, shorts blocked, nothing to close)
    becomes a hold. Reward = daily change in account equity as a percent of starting capital.
    """
    S, iv, rf, dates = md['S'], md['iv'], md['rf'], md['dates']
    short_allowed = md.get('short_allowed')
    tenor, mult, cap = cfg['tenor_days'], cfg['contract_mult'], cfg['capital']
    max_pos = cfg['max_positions']
    max_short = cfg.get('max_short_positions', max_pos)
    cost, m = cfg['cost_vol'], cfg.get('moneyness', 0.0)
    m_pct, m_min = cfg.get('short_margin_pct', 0.20), cfg.get('min_margin_pct', 0.10)
    n = len(S)

    def value(p, t, vol_shift=0.0):
        tau = max(p['expiry_t'] - t, 0) / TRADING_DAYS
        kind = p['Type'].lower()
        if tau == 0:
            return max(S[t] - p['Strike'], 0) if kind == 'call' else max(p['Strike'] - S[t], 0)
        return float(bs_price(S[t], p['Strike'], tau, rf[t], max(iv[t] + vol_shift, 0.01), 0.0, kind))

    def margin_used(t):
        return sum(short_margin_requirement(p['Type'].lower(), S[t], p['Strike'], value(p, t), mult, m_pct, m_min)
                   for p in book if p['sign'] < 0)

    def mark(t):
        return cash + sum(p['sign'] * value(p, t) * mult for p in book)

    def close(p, t, reason, use_spread=True):
        # Longs sell at the bid, shorts buy back at the ask; expiry and end-of-data use intrinsic / mid
        exit_value = value(p, t, -p['sign'] * cost if use_spread else 0.0)
        sign = p['sign']
        pnl = sign * (exit_value - p['Entry Premium']) * mult
        ret = 100 * sign * (exit_value - p['Entry Premium']) / p['Entry Premium']
        trades.append({**{k: v for k, v in p.items() if k not in ('entry_t', 'expiry_t', 'sign')},
                       'Exit Date': dates[t], 'Exit Reason': reason, 'Exit Spot': S[t], 'Exit Value': exit_value,
                       'P&L ($)': pnl, 'Return %': ret, 'Holding Days': t - p['entry_t']})
        exits.append(t)
        if verbose:
            print(f"{dates[t]:%Y-%m-%d} close {p['Position']} K={p['Strike']:.0f} ({reason}) P&L {format_price(pnl)}")
        return sign * exit_value * mult           # cash flow: +proceeds for longs, -buy-back cost for shorts

    book, trades, losses, exits = [], [], [], []
    actions = np.zeros(n, dtype=int)
    entries = {name: [] for name in POSITION_ORDER}
    counts = {name: np.zeros(n, dtype=int) for name in POSITION_ORDER}
    cash = float(cap)
    equity, margin = np.full(n, np.nan), np.zeros(n)
    equity[0] = cap
    state = np.concatenate([market_states[0], _book_features(book, 0, max_pos, tenor, 0.0, cap)])

    for t in progress(range(n - 1), desc=desc):
        action = policy(t, state) if policy is not None else agent.act(state)
        actions[t] = action

        if action in _OPEN_ACTIONS and len(book) < max_pos:
            kind, sign = _OPEN_ACTIONS[action]
            strike = S[t] * (1 + m) if kind == 'call' else S[t] * (1 - m)
            T = tenor / TRADING_DAYS
            premium = float(bs_price(S[t], strike, T, rf[t], max(iv[t] + sign * cost, 0.01), 0.0, kind))
            free_equity = mark(t) - margin_used(t)
            if sign > 0:
                ok = free_equity >= premium * mult
            else:
                n_short = sum(p['sign'] < 0 for p in book)
                allowed = True if short_allowed is None else bool(short_allowed[t])
                new_margin = short_margin_requirement(kind, S[t], strike, premium, mult, m_pct, m_min)
                ok = allowed and n_short < max_short and free_equity >= new_margin
            if ok:
                greeks = bs_greeks(S[t], strike, T, rf[t], iv[t], 0.0, kind)
                side = 'Long' if sign > 0 else 'Short'
                p = {'Position': f'{side} {kind.title()}', 'Type': kind.title(), 'Side': side,
                     'Entry Date': dates[t], 'Expiry Date': dates[min(t + tenor, n - 1)], 'Strike': strike,
                     'Entry Spot': S[t], 'Entry IV': iv[t], 'Forecast Vol': md['sigma_f'][t],
                     'GBM Edge %': 100 * md[f'{kind}_edge'][t], 'Entry Premium': premium,
                     # position Greeks (signed: a short call has negative delta)
                     'Delta': sign * float(greeks['delta']), 'Gamma': sign * float(greeks['gamma']),
                     'Vega': sign * float(greeks['vega']), 'Theta': sign * float(greeks['theta']),
                     'sign': sign, 'entry_t': t, 'expiry_t': t + tenor}
                book.append(p)
                cash -= sign * premium * mult      # pay for longs, collect for shorts
                entries[p['Position']].append(t)
                if verbose:
                    print(f"{dates[t]:%Y-%m-%d} {side.lower()} {kind} K={strike:.0f} @ {format_price(premium)}")
        elif action == 5 and book:
            cash += close(book.pop(0), t, 'Agent close')

        # ---- move to t+1: expiries, mark-to-market, margin calls ----
        t1 = t + 1
        for p in [p for p in book if p['expiry_t'] <= t1]:
            book.remove(p)
            cash += close(p, t1, 'Expired', use_spread=False)
        while any(p['sign'] < 0 for p in book) and mark(t1) < margin_used(t1):
            oldest_short = next(p for p in book if p['sign'] < 0)
            book.remove(oldest_short)
            cash += close(oldest_short, t1, 'Margin call')

        equity[t1], margin[t1] = mark(t1), margin_used(t1)
        for name in POSITION_ORDER:
            counts[name][t1] = sum(p['Position'] == name for p in book)

        reward = 100 * (equity[t1] - equity[t]) / cap
        next_state = np.concatenate([market_states[t1],
                                     _book_features(book, t1, max_pos, tenor, margin[t1], equity[t1])])
        if train and agent is not None:
            agent.remember(state, action, reward, next_state, t1 == n - 1)
            if len(agent.memory) > batch_size and t % train_every == 0:
                losses.append(agent.exp_replay(batch_size))
        state = next_state

    open_positions = len(book)
    for p in list(book):                                     # close what is left at the last mark (mid)
        book.remove(p)
        close(p, n - 1, 'End of data', use_spread=False)

    return {'equity': pd.Series(equity, index=dates, name='Equity'), 'trades': pd.DataFrame(trades),
            'losses': losses, 'actions': actions, 'entries': entries, 'exits': exits,
            'counts': {k: pd.Series(v, index=dates) for k, v in counts.items()},
            'margin': pd.Series(margin, index=dates, name='Margin used'), 'open_positions': open_positions}


def options_trade_statistics(trades, groups=('Side', 'Type', 'Position')):
    """Trade statistics for All trades and per Side (Long/Short), Type (Call/Put) and Position."""
    buckets = {'All': trades}
    if len(trades):
        order = ['Long', 'Short', 'Call', 'Put'] + POSITION_ORDER
        found = {k: g for col in groups if col in trades for k, g in trades.groupby(col)}
        buckets.update({k: found[k] for k in order if k in found})
    rows = {}
    for name, g in buckets.items():
        if not len(g):
            rows[name] = {'Trades': 0}
            continue
        pnl = g['P&L ($)']
        gross_loss = -pnl[pnl <= 0].sum()
        rows[name] = {
            'Trades': len(g),
            'Win Rate': (pnl > 0).mean(),
            'Total P&L ($)': pnl.sum(),
            'Avg P&L ($)': pnl.mean(),
            'Avg Return %': g['Return %'].mean(),
            'Median Return %': g['Return %'].median(),
            'Best Trade ($)': pnl.max(),
            'Worst Trade ($)': pnl.min(),
            'Profit Factor': pnl[pnl > 0].sum() / gross_loss if gross_loss else np.nan,
            'Avg Holding Days': g['Holding Days'].mean(),
            'Held to Expiry': (g['Exit Reason'] == 'Expired').mean(),
            'Margin Calls': int((g['Exit Reason'] == 'Margin call').sum()),
            'Avg Entry IV': g['Entry IV'].mean(),
            'Avg GBM Edge %': g['GBM Edge %'].mean(),
        }
    return pd.DataFrame(rows)


def print_options_summary(result, title):
    stats = options_trade_statistics(result['trades'])
    eq = result['equity']
    print('-' * 70)
    print(title)
    print(f"Final equity: {format_price(eq.iloc[-1])} ({100 * (eq.iloc[-1] / eq.iloc[0] - 1):+.1f}%) | "
          f"open at end: {result['open_positions']} | peak margin used: {format_price(result['margin'].max())}")
    for name in ['All'] + POSITION_ORDER:
        if name in stats and stats[name]['Trades']:
            s = stats[name]
            print(f"{name:>10}: {int(s['Trades']):4d} trades | win rate {s['Win Rate']:.1%} | "
                  f"P&L {format_price(s['Total P&L ($)'])} | avg return {s['Avg Return %']:+.1f}%"
                  + (f" | margin calls {int(s['Margin Calls'])}" if s['Margin Calls'] else ''))
    if result['losses']:
        print(f"Max batch loss: {max(result['losses']):.4f}")
    print('-' * 70)


def plot_options_behavior(result, S, title):
    """Spot with entries/exits for all four position types, open positions, margin and account equity."""
    dates = result['equity'].index
    S = pd.Series(np.asarray(S), index=dates)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48, 0.17, 0.12, 0.23],
                        vertical_spacing=0.035,
                        subplot_titles=('Underlying with option entries / exits',
                                        'Open positions (longs above zero, shorts below)', 'Margin used',
                                        'Account equity'))
    fig.add_scatter(x=dates, y=S, name='Spot', mode='lines', line=dict(color='#e6e6e6', width=1.3), row=1, col=1)
    for name in POSITION_ORDER:
        idx = result['entries'][name]
        color, symbol = POSITION_STYLE[name]
        if idx:
            fig.add_scatter(x=dates[idx], y=S.iloc[idx], name=name, legendgroup=name, mode='markers',
                            marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1.5, color=color)),
                            row=1, col=1)
        sign = 1 if name.startswith('Long') else -1
        fig.add_bar(x=dates, y=sign * result['counts'][name], name=name, legendgroup=name, showlegend=not idx,
                    marker_color=color, opacity=0.75, row=2, col=1)
    if result['exits']:
        fig.add_scatter(x=dates[result['exits']], y=S.iloc[result['exits']], name='Exit / expiry', mode='markers',
                        marker=dict(symbol='x-thin', size=7, color=PALETTE[3], line=dict(width=1.5, color=PALETTE[3])),
                        row=1, col=1)
    fig.add_scatter(x=dates, y=result['margin'], name='Margin used', mode='lines', showlegend=False,
                    line=dict(color=PALETTE[4], width=1.2), fill='tozeroy', fillcolor=_rgba(PALETTE[4], 0.15),
                    row=3, col=1)
    eq = result['equity']
    fig.add_scatter(x=dates, y=eq, name='Equity', mode='lines', showlegend=False,
                    line=dict(color=PALETTE[0], width=2), fill='tozeroy', fillcolor=_rgba(PALETTE[0], 0.08),
                    row=4, col=1)
    fig.update_yaxes(tickprefix='$', row=3, col=1)
    fig.update_yaxes(range=[eq.min() * 0.97, eq.max() * 1.03], tickprefix='$', row=4, col=1)
    fig.update_layout(title=subtitled(title, f'Final equity {format_price(eq.iloc[-1])}'), height=940,
                      barmode='relative', bargap=0)
    fig.show()


def options_trade_analysis_figure(trades, title='Options Trade Analysis'):
    """2x2 dashboard: cumulative P&L by position, return distribution, return vs GBM edge, P&L by exit reason."""
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=('Cumulative realized P&L by position', 'Trade return distribution',
                                        'Trade return vs GBM edge at entry', 'P&L by exit reason and position'))
    if len(trades):
        ordered = trades.sort_values('Exit Date')
        fig.add_scatter(x=ordered['Exit Date'], y=ordered['P&L ($)'].cumsum(), name='All', mode='lines',
                        line=dict(color='#e6e6e6', width=2.4), row=1, col=1)
        for name in POSITION_ORDER:
            g = ordered[ordered['Position'] == name]
            if not len(g):
                continue
            color = POSITION_STYLE[name][0]
            fig.add_scatter(x=g['Exit Date'], y=g['P&L ($)'].cumsum(), name=name, legendgroup=name,
                            mode='lines', line=dict(color=color, width=1.6), row=1, col=1)
            fig.add_histogram(x=g['Return %'], name=name, legendgroup=name, showlegend=False,
                              marker_color=color, opacity=0.55, nbinsx=40, row=1, col=2)
            fig.add_scatter(x=g['GBM Edge %'], y=g['Return %'], mode='markers', name=name, legendgroup=name,
                            showlegend=False, marker=dict(color=color, size=6, opacity=0.6), row=2, col=1)
        by_reason = trades.groupby(['Exit Reason', 'Position'])['P&L ($)'].sum().reset_index()
        fig.add_bar(x=by_reason['Exit Reason'] + ' · ' + by_reason['Position'], y=by_reason['P&L ($)'],
                    marker_color=[POSITION_STYLE.get(k, (NEUTRAL,))[0] for k in by_reason['Position']],
                    showlegend=False, hovertemplate='%{x}<br>%{y:$,.0f}<extra></extra>', row=2, col=2)
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1), row=2, col=1)
    fig.update_xaxes(title_text='Return % (shorts: % of premium collected)', row=1, col=2)
    fig.update_xaxes(title_text='GBM edge at entry (%)', row=2, col=1)
    fig.update_yaxes(title_text='$', row=1, col=1)
    fig.update_yaxes(title_text='Return %', row=2, col=1)
    fig.update_yaxes(title_text='$', row=2, col=2)
    fig.update_layout(title=title, height=800, barmode='overlay', hovermode='closest')
    return fig
