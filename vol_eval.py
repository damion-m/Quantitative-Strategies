"""Tools for judging volatility forecasts (used by volatility_forecasts_revisited.ipynb).

Plain functions on pandas Series indexed by date; written in the shape `func_lib.py`'s volatility
section will take when it is split out. Variances are DAILY variances (annualize with x 252).

    qlike, squared_error     per-day losses of a variance forecast
    newey_west_variance      long-run variance of a serially correlated series
    diebold_mariano          is one forecast's loss reliably lower than another's?
    mincer_zarnowitz         is a forecast unbiased and correctly scaled?
    expanding_percentile     where today's value sits among all values SO FAR (no look-ahead)
    debias                   rescale a forecast by its own past ratio of outcome to forecast
    walk_forward_blend       regression combination of forecasts, refitted on past data only
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

TRADING_DAYS = 252


def qlike(forecast: pd.Series, realized: pd.Series) -> pd.Series:
    """QLIKE loss, y/f - log(y/f) - 1 (Patton 2011). Zero when the forecast is exact. It penalizes
    UNDER-forecasting more than over-forecasting, and it still ranks forecasts correctly when the
    'realized' variance is itself a noisy estimate, which squared error in volatility does not."""
    ratio = realized / forecast.where(forecast > 0)
    return (ratio - np.log(ratio) - 1).where(realized > 0)


def squared_error(forecast: pd.Series, realized: pd.Series) -> pd.Series:
    """Squared error in annualized variance units (dominated by crisis days; shown for contrast)."""
    return ((forecast - realized) * TRADING_DAYS) ** 2


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of x with Bartlett weights: gamma_0 + 2 sum_k (1 - k/(L+1)) gamma_k.
    Needed because 21-day targets on neighbouring days share 20 days, so daily losses are not independent."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(max(v, 1e-300))


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int) -> dict:
    """Diebold-Mariano test of equal accuracy. d_t = loss_a - loss_b; DM = mean(d) / sqrt(LRV(d) / n).
    Negative DM: forecast A has the lower loss. p is two-sided under a normal distribution."""
    d = (loss_a - loss_b).dropna().to_numpy()
    stat = d.mean() / np.sqrt(newey_west_variance(d, lags) / len(d))
    return {'dm': float(stat), 'p': float(2 * norm.sf(abs(stat))), 'n': int(len(d))}


def mincer_zarnowitz(forecast: pd.Series, realized: pd.Series, lags: int) -> dict:
    """Regress realized = a + b * forecast. An unbiased, correctly scaled forecast has a = 0, b = 1.
    Standard errors are Newey-West; p_joint tests a = 0 and b = 1 together (Wald, chi-squared with 2 df).
    b < 1 means the forecast over-reacts: when it is high, the outcome is lower than it says."""
    ok = forecast.notna() & realized.notna()
    f, y = forecast[ok].to_numpy() * TRADING_DAYS, realized[ok].to_numpy() * TRADING_DAYS
    X = np.column_stack([np.ones(len(f)), f])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ beta
    n = len(y)
    g = X * u[:, None]                                     # moment contributions
    S = g.T @ g / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = g[k:].T @ g[:-k] / n
        S += w * (G + G.T)
    bread = np.linalg.inv(X.T @ X / n)
    cov = bread @ S @ bread / n
    r = beta - np.array([0.0, 1.0])
    wald = float(r @ np.linalg.solve(cov, r))
    return {'a': float(beta[0]), 'b': float(beta[1]), 'se_b': float(np.sqrt(cov[1, 1])),
            'r2': float(1 - u.var() / y.var()), 'p_joint': float(chi2.sf(wald, 2)), 'n': int(n)}


def expanding_percentile(x: pd.Series, min_periods: int = 252) -> pd.Series:
    """Percentile of each value among all values up to and including that day."""
    return x.expanding(min_periods).rank(pct=True)


def debias(forecast: pd.Series, realized_fwd: pd.Series, horizon: int, min_periods: int = 252) -> pd.Series:
    """forecast x (average past ratio of outcome to forecast). The ratio for day s is known only at
    s + horizon, when its outcome window has closed, so the average is lagged by `horizon` days."""
    ratio = (realized_fwd / forecast.where(forecast > 0)).shift(horizon)
    return forecast * ratio.expanding(min_periods).mean()


def walk_forward_blend(forecasts: Dict[str, pd.Series], realized_fwd: pd.Series, first_date: str, horizon: int,
                       refit_freq: str = 'YS') -> pd.Series:
    """Least-squares combination realized ~ a + sum_i b_i forecast_i, refitted at each `refit_freq`
    boundary on rows whose outcome window had closed by then. Returns out-of-sample blended forecasts."""
    X = pd.DataFrame(forecasts)
    idx = X.index
    starts = sorted({idx.searchsorted(d) for d in [pd.Timestamp(first_date), *pd.date_range(first_date, idx[-1], freq=refit_freq)]
                     if idx.searchsorted(d) < len(idx)})
    out = pd.Series(np.nan, index=idx)
    for a, b in zip(starts, starts[1:] + [len(idx)]):
        tr = slice(0, max(a - horizon, 0))
        ok = X.iloc[tr].notna().all(axis=1) & realized_fwd.iloc[tr].notna()
        if ok.sum() < 250:
            continue
        A = np.column_stack([np.ones(ok.sum()), X.iloc[tr][ok].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, realized_fwd.iloc[tr][ok].to_numpy(), rcond=None)
        out.iloc[a:b] = coef[0] + X.iloc[a:b].to_numpy() @ coef[1:]
    floor = float(np.nanpercentile(realized_fwd.dropna(), 1))
    return out.clip(lower=floor)
