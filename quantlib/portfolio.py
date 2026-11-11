"""Section 6 of the original func_lib, portfolio part: performance statistics, the stats
table figure, risk parity weights and portfolio backtests.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import TRADING_DAYS
from .theme import PALETTE, RANGE_SELECTOR, has_subtitle


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
