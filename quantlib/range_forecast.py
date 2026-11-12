"""A forecast of price as a range (price_range_forecast.ipynb).

Plain functions on dates x tickers frames; written in the shape `func_lib.py`'s volatility and
time-series sections will take when split out.

    walk_forward_vol_model     pooled regression of next-month log volatility on today's features,
                               refitted each January on past data only
    walk_forward_centre        the market's expected next-month return given the VIX state, fitted
                               on past data only
    past_quantiles             empirical quantiles of standardized outcomes, from the past only
    bands                      lower and upper price-return bounds for a coverage level
    score_bands                coverage, width and interval score
    sized_returns              a position scaled down when predicted volatility is high
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NORMAL_Z = {0.05: -1.6449, 0.25: -0.6745, 0.75: 0.6745, 0.95: 1.6449}


def refit_starts(index: pd.DatetimeIndex, first: str, freq: str = 'YS') -> List[Tuple[int, int]]:
    """(start, end) row positions of each block between refits, the first beginning at `first`."""
    starts = sorted({index.searchsorted(d) for d in [pd.Timestamp(first), *pd.date_range(first, index[-1], freq=freq)]
                     if index.searchsorted(d) < len(index)})
    return list(zip(starts, starts[1:] + [len(index)]))


def walk_forward_vol_model(features: Dict[str, pd.DataFrame], fwd_vol: pd.DataFrame, member: pd.DataFrame, first: str,
                           horizon: int = 21, sample_every: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predicted next-`horizon`-day annualized volatility for every stock and day from `first` on.

    Model: log(vol over the next month) = a + sum_k b_k x_k, one set of weights for all stocks,
    fitted by least squares on member stock-days whose outcome window had closed before the refit
    date (every `sample_every`-th day: neighbouring days' outcomes overlap). Logs make the errors
    roughly symmetric and keep every prediction positive. Returns (predictions, weights by refit date).
    """
    names = list(features)
    idx, cols = member.index, member.columns
    X = np.stack([features[k].reindex(index=idx, columns=cols).to_numpy(float) for k in names], axis=2)
    y = np.log(fwd_vol.reindex(index=idx, columns=cols).where(lambda v: v > 0)).to_numpy(float)
    ok = np.isfinite(X).all(axis=2) & member.to_numpy()
    pred = np.full(y.shape, np.nan)
    coefs = {}
    for a, b in refit_starts(idx, first):
        rows = np.arange(0, max(a - horizon, 0), sample_every)
        m = ok[rows] & np.isfinite(y[rows])
        if m.sum() < 5000:
            continue
        A = np.column_stack([np.ones(m.sum()), X[rows][m]])
        beta, *_ = np.linalg.lstsq(A, y[rows][m], rcond=None)
        block = X[a:b]
        pred[a:b] = np.where(ok[a:b], beta[0] + block @ beta[1:], np.nan)
        coefs[idx[a]] = dict(zip(['const'] + names, beta))
    return pd.DataFrame(np.exp(pred), index=idx, columns=cols), pd.DataFrame(coefs).T


def walk_forward_centre(state: pd.Series, market_fwd: pd.Series, first: str, horizon: int = 21) -> Tuple[pd.Series, pd.Series]:
    """Two forecasts of the market's next-`horizon`-day log return, each using the past only:
    `flat` = the average of all past outcomes; `tilted` = a + b * state from a least-squares fit on
    past data. Refitted each January."""
    idx = state.index
    flat, tilted = pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
    for a, b in refit_starts(idx, first):
        tr = slice(0, max(a - horizon, 0))
        d = pd.concat([state.iloc[tr], market_fwd.iloc[tr]], axis=1).dropna()
        if len(d) < 250:
            continue
        A = np.column_stack([np.ones(len(d)), d.iloc[:, 0].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, d.iloc[:, 1].to_numpy(), rcond=None)
        flat.iloc[a:b] = d.iloc[:, 1].mean()
        tilted.iloc[a:b] = beta[0] + beta[1] * state.iloc[a:b].to_numpy()
    return flat, tilted


def past_quantiles(z: pd.DataFrame, first: str, horizon: int = 21, probs: Iterable[float] = (0.05, 0.25, 0.75, 0.95)) -> pd.DataFrame:
    """For each refit date, the quantiles of the standardized outcomes z (pooled over all stocks)
    whose windows had closed by then. Index: every date from `first`; columns: the probabilities."""
    out = pd.DataFrame(np.nan, index=z.index, columns=list(probs))
    v = z.to_numpy(float)
    for a, b in refit_starts(z.index, first):
        past = v[:max(a - horizon, 0)]
        past = past[np.isfinite(past)]
        if len(past) < 5000:
            continue
        out.iloc[a:b] = np.quantile(past, list(probs))
    return out


def bands(sigma: pd.DataFrame, centre, q_lo, q_hi, horizon: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bounds for the next-`horizon`-day log return: centre + q * sigma * sqrt(horizon / 252).
    `centre`, `q_lo`, `q_hi` may be numbers or Series indexed by date."""
    s = sigma * np.sqrt(horizon / TRADING_DAYS)
    c = centre if np.isscalar(centre) else pd.Series(centre).reindex(sigma.index)
    lo = s.mul(q_lo, axis=0) if not np.isscalar(q_lo) else s * q_lo
    hi = s.mul(q_hi, axis=0) if not np.isscalar(q_hi) else s * q_hi
    if np.isscalar(c):
        return lo + c, hi + c
    return lo.add(c, axis=0), hi.add(c, axis=0)


def score_bands(lo: pd.DataFrame, hi: pd.DataFrame, outcome: pd.DataFrame, nominal: float) -> dict:
    """coverage: share of outcomes inside [lo, hi] (should equal `nominal`).
    width: average hi - lo, in log-return units.
    interval_score (Gneiting and Raftery 2007): width + (2 / alpha) * distance outside the band,
    alpha = 1 - nominal. It rewards narrow bands and punishes misses; lower is better, and a band
    cannot win by being wide or by being narrow alone."""
    ok = lo.notna() & hi.notna() & outcome.notna()
    l, h, y = lo.where(ok).to_numpy(), hi.where(ok).to_numpy(), outcome.where(ok).to_numpy()
    alpha = 1 - nominal
    width = h - l
    miss = np.where(y < l, l - y, 0.0) + np.where(y > h, y - h, 0.0)
    return {'coverage': float(np.nanmean((y >= l) & (y <= h)) / np.nanmean(ok.to_numpy()) if ok.to_numpy().any() else np.nan),
            'width': float(np.nanmean(width)), 'interval_score': float(np.nanmean(width + 2 / alpha * miss)), 'n': int(ok.to_numpy().sum())}


def sized_returns(ret: pd.DataFrame, sigma: pd.DataFrame, target: pd.DataFrame, every: int = 5, cost_bps: float = 5.0,
                  cap: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Each stock held at exposure min(cap, target / sigma), reset every `every` days from the
    volatility known at the previous close and held in between. No leverage with cap = 1: the rule
    only ever reduces a position. Cost: cost_bps on every change of exposure.
    Returns (daily net returns, exposures)."""
    w = (target / sigma).clip(upper=cap)
    keep = pd.Series(np.arange(len(w)) % every == 0, index=w.index)
    w = w.where(keep, axis=0).ffill().shift(1)                 # decided at the close, earns from the next day
    cost = w.diff().abs().fillna(0.0) * cost_bps / 1e4
    return w * ret - cost, w


def perf_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Per column: annual return, volatility, Sharpe ratio (no risk-free rate), worst drawdown."""
    r = returns.dropna(how='all')
    eq = (1 + r.fillna(0.0)).cumprod()
    sd = r.std(ddof=1)
    return pd.DataFrame({'ann_return': r.mean() * TRADING_DAYS, 'ann_vol': sd * np.sqrt(TRADING_DAYS),
                         'sharpe': r.mean() / sd.where(sd > 0) * np.sqrt(TRADING_DAYS), 'max_drawdown': (1 - eq / eq.cummax()).max()})
