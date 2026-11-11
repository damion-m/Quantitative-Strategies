"""Section 3 of the original func_lib: time-series features and charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import TRADING_DAYS
from .theme import DOWN, PALETTE, RANGE_SELECTOR, UP, _rgba


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
