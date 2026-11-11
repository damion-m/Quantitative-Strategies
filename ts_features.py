"""Tests of whether price-derived features predict anything (time_series_features_revisited.ipynb).

Plain functions; written in the shape `func_lib.py`'s time-series section will take when split out.

    stationary_features     the seven dollar-level features re-expressed as ratios that mean the
                            same thing at any price
    effective_count         how many independent pieces of information a set of features holds
    predictive_table        rank correlation of each feature with a later outcome, with a t-statistic
                            that allows for overlapping horizons
    wide_features           the same stationary features for every stock in a dates x tickers panel
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from vol_eval import newey_west_variance

TRADING_DAYS = 252


def stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """From the original columns (Close, SMA_20, SMA_50, Volatility, BB_upper, BB_lower, RSI,
    Return_Volatility) to scale-free versions:

        dist_sma20    Close / SMA_20 - 1          how far above its 20-day average
        dist_sma50    Close / SMA_50 - 1
        sma_spread    SMA_20 / SMA_50 - 1         the trend gap
        band_pos      (Close - BB_lower) / (BB_upper - BB_lower)    0 = lower band, 1 = upper band
        band_width    (BB_upper - BB_lower) / SMA_20                 width relative to price
        rsi           RSI / 100
        ret_vol       annualized 20-day return volatility
    """
    return pd.DataFrame({
        'dist_sma20': df['Close'] / df['SMA_20'] - 1,
        'dist_sma50': df['Close'] / df['SMA_50'] - 1,
        'sma_spread': df['SMA_20'] / df['SMA_50'] - 1,
        'band_pos': (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']),
        'band_width': (df['BB_upper'] - df['BB_lower']) / df['SMA_20'],
        'rsi': df['RSI'] / 100,
        'ret_vol': df['Return_Volatility'],
    })


def effective_count(x: pd.DataFrame) -> dict:
    """Eigenvalues of the correlation matrix. `share_first` is the fraction of all variation carried
    by the single strongest combination; `effective` = (sum lambda)^2 / sum lambda^2 is the number of
    equally important independent features that would look the same (the participation ratio)."""
    lam = np.linalg.eigvalsh(x.dropna().corr().to_numpy())[::-1]
    return {'share_first': float(lam[0] / lam.sum()), 'effective': float(lam.sum() ** 2 / (lam ** 2).sum()), 'features': int(len(lam))}


def predictive_table(features: pd.DataFrame, outcome: pd.Series, horizon: int) -> pd.DataFrame:
    """Spearman correlation of each feature today with `outcome` (already aligned to today), and a
    t-statistic from the regression of the outcome's rank on the feature's rank with Newey-West
    errors (lags = horizon - 1, because h-day outcomes on neighbouring days overlap)."""
    rows = {}
    for k in features:
        d = pd.concat([features[k], outcome], axis=1).dropna()
        x, y = d.iloc[:, 0].rank().to_numpy(), d.iloc[:, 1].rank().to_numpy()
        x, y = x - x.mean(), y - y.mean()
        b = float(x @ y / (x @ x))
        u = y - b * x
        se = np.sqrt(newey_west_variance(x * u, max(horizon - 1, 0)) * len(x) / (x @ x) ** 2)
        rows[k] = {'corr': float(np.corrcoef(x, y)[0, 1]), 't': float(b / se), 'days': int(len(d))}
    return pd.DataFrame(rows).T


def wilder_rsi_wide(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI for every column at once (same smoothing as func_lib.wilder_rsi)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def wide_features(close: pd.DataFrame, sma_short: int = 20, sma_long: int = 50, bb_k: float = 2.0,
                  rsi_window: int = 14) -> Dict[str, pd.DataFrame]:
    """The stationary features for every stock in a dates x tickers frame of closes."""
    s20, s50 = close.rolling(sma_short).mean(), close.rolling(sma_long).mean()
    sd = close.rolling(sma_short).std()
    upper, lower = s20 + bb_k * sd, s20 - bb_k * sd
    return {'dist_sma20': close / s20 - 1, 'dist_sma50': close / s50 - 1, 'sma_spread': s20 / s50 - 1,
            'band_pos': (close - lower) / (upper - lower), 'band_width': (upper - lower) / s20,
            'rsi': wilder_rsi_wide(close, rsi_window) / 100,
            'ret_vol': close.pct_change(fill_method=None).rolling(sma_short).std() * np.sqrt(TRADING_DAYS)}


# ----------------------------------------------------------------------------------------
# Round 2: eight new candidate features (risk state, market state, events, relative)
# ----------------------------------------------------------------------------------------
def daily_variance_wide(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Range-based daily variance for every stock: overnight gap^2 + the Garman-Klass intraday
    estimator, 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2. Uses the whole day's range, so it is several
    times less noisy than a squared close-to-close return (same formula as func_lib.daily_variance_proxy)."""
    overnight = np.log(open_ / close.shift(1)) ** 2
    return overnight + 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2


def risk_features(open_, high, low, close, window: int = 20) -> Dict[str, pd.DataFrame]:
    """gk_vol        annualized volatility from the 20-day average range-based variance
    downside_vol  annualized volatility counting only down days (semi-deviation)
    vol_of_vol    63-day standard deviation of daily changes in log 20-day volatility"""
    r = np.log(close).diff()
    gk = np.sqrt(daily_variance_wide(open_, high, low, close).rolling(window).mean().clip(lower=0) * TRADING_DAYS)
    down = np.sqrt((r.clip(upper=0) ** 2).rolling(window).mean() * TRADING_DAYS)
    vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    vov = np.log(vol.where(vol > 0)).diff().rolling(63).std()
    return {'gk_vol': gk, 'downside_vol': down, 'vol_of_vol': vov}


def relative_return(close: pd.DataFrame, sectors: pd.Series, member: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """The stock's return over `window` days minus the average for its sector's members."""
    r = close.pct_change(window, fill_method=None).where(member)
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for _, cols in sectors.dropna().groupby(sectors.dropna()).groups.items():
        cols = [c for c in cols if c in r.columns]
        out[cols] = r[cols].sub(r[cols].mean(axis=1), axis=0)
    return out


def event_features(events: pd.DataFrame, index: pd.DatetimeIndex, columns, life: int = 63) -> Dict[str, pd.DataFrame]:
    """From a table of earnings announcements (ticker, announced, sue):

    sue                  the latest earnings surprise, usable from the first trading day AFTER the
                         announcement and for `life` trading days (about one quarter), then missing
    days_since_earnings  trading days since that first usable day (missing after `life`)
    """
    sue = pd.DataFrame(np.nan, index=index, columns=columns)
    age = pd.DataFrame(np.nan, index=index, columns=columns)
    for t, g in events.dropna(subset=['sue']).sort_values('announced').groupby('ticker'):
        if t not in sue.columns:
            continue
        pos = index.searchsorted(pd.DatetimeIndex(g['announced']), side='right')        # strictly after the announcement day
        s = np.full(len(index), np.nan)
        a = np.full(len(index), np.nan)
        for p, v in zip(pos, g['sue'].to_numpy()):
            if p < len(index):
                s[p:p + life] = v
                a[p:p + life] = np.arange(min(life, len(index) - p))
        sue[t], age[t] = s, a
    return {'sue': sue, 'days_since_earnings': age}


def market_state(vix: pd.Series, vix3m: pd.Series, min_periods: int = 252) -> pd.DataFrame:
    """vix_pct    where today's VIX sits among all VIX readings so far (0 to 1)
    vix_slope  VIX3M / VIX - 1: positive in calm markets (later months priced higher), negative in stress"""
    return pd.DataFrame({'vix_pct': vix.expanding(min_periods).rank(pct=True), 'vix_slope': vix3m / vix - 1})


def benjamini_hochberg(p: pd.Series, q: float = 0.10) -> pd.Series:
    """True for the hypotheses rejected by the Benjamini-Hochberg step-up rule at false-discovery rate q."""
    s = p.dropna().sort_values()
    below = s.to_numpy() <= np.arange(1, len(s) + 1) / len(s) * q
    k = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    return pd.Series(p.index.isin(s.index[:k]), index=p.index)
