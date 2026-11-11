"""Section 6 of the original func_lib, volatility part: variance proxy, GARCH(1,1),
HAR-RV and forecast metrics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from ._common import TRADING_DAYS


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


# ---- Forecaster classes: fit() / forecast() around the functions above ----
class GarchForecaster:
    """GARCH(1,1) forecaster of the average daily variance over the next `horizon` days.

    A thin wrapper around fit_garch11 and garch_forecast that keeps the fitted parameters:
        model = GarchForecaster(horizon=21).fit(log_ret[:'2020'])
        model.params                         # dict(mu, omega, alpha, beta, h0), as fit_garch11 returns
        model.forecast(log_ret)              # Series, as garch_forecast returns
        model.walk_forward(log_ret, '2015-01-01')   # same as walk_forward_garch(log_ret, '2015-01-01', 21)
    """
    name = 'GARCH'

    def __init__(self, horizon=21):
        self.horizon = horizon
        self.params = None

    def fit(self, returns):
        """Gaussian MLE on daily returns (fit_garch11). Returns self."""
        self.params = fit_garch11(returns)
        return self

    def forecast(self, returns, horizon=None):
        """Average daily variance over t+1..t+horizon for every t, filtered through `returns` (garch_forecast)."""
        if self.params is None:
            raise RuntimeError('call fit() before forecast()')
        return garch_forecast(returns, self.params, self.horizon if horizon is None else horizon)

    def walk_forward(self, returns, first_test_date, refit_freq='YS'):
        """Out-of-sample forecasts, refitted on all returns before each refit date.

        Gives exactly what walk_forward_garch(returns, first_test_date, self.horizon, refit_freq) gives.
        Afterwards self.params holds the last fit.
        """
        forecasts = pd.Series(np.nan, index=returns.index, name=self.name)
        clean = returns.fillna(0.0)
        for start, end in _refit_blocks(returns.index, first_test_date, refit_freq):
            self.fit(clean.iloc[:start])
            forecasts.iloc[start:end] = self.forecast(clean).iloc[start:end].to_numpy()
        return forecasts


class HarForecaster:
    """HAR-RV forecaster (Corsi 2009): OLS of the target on daily, weekly and monthly average realized
    variance (har_design), keeping the fitted coefficients.

        model = HarForecaster(horizon=21).fit(rv[:'2020'], target[:'2020'])
        model.coef                           # [intercept, rv_d, rv_w, rv_m]
        model.forecast(rv)                   # Series of forecasts for every day with a full design row
        model.walk_forward(rv, target, '2015-01-01')   # same as walk_forward_har(rv, target, '2015-01-01', 21)
    """
    name = 'HAR'

    def __init__(self, horizon=21):
        self.horizon = horizon
        self.coef = None

    @staticmethod
    def _ols(X, y):
        ok = X.notna().all(axis=1) & y.notna()
        A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        return coef

    def fit(self, rv, target):
        """OLS on every row where the HAR regressors and the target are all known. Returns self.

        It uses all the rows it is given: to stay out of sample, pass only targets whose window
        (t+1..t+horizon) has ended, as walk_forward does.
        """
        self.coef = self._ols(har_design(rv), target)
        return self

    def forecast(self, rv, floor=None):
        """coef[0] + har_design(rv) @ coef[1:] for every row (NaN during the 22-day warm-up),
        clipped below at `floor` when given."""
        if self.coef is None:
            raise RuntimeError('call fit() before forecast()')
        X = har_design(rv)
        forecasts = pd.Series(self.coef[0] + X.to_numpy() @ self.coef[1:], index=rv.index, name=self.name)
        return forecasts if floor is None else forecasts.clip(lower=floor)

    def walk_forward(self, rv, target, first_test_date, refit_freq='YS'):
        """Out-of-sample forecasts, refitted at each `refit_freq` boundary on data known by then.

        Gives exactly what walk_forward_har(rv, target, first_test_date, self.horizon, refit_freq)
        gives, including the floor at the 1st percentile of rv. Afterwards self.coef holds the last fit.
        """
        X = har_design(rv)
        forecasts = pd.Series(np.nan, index=rv.index, name=self.name)
        for start, end in _refit_blocks(rv.index, first_test_date, refit_freq):
            train = slice(0, max(start - self.horizon, 0))
            self.coef = self._ols(X.iloc[train], target.iloc[train])
            Xp = X.iloc[start:end]
            forecasts.iloc[start:end] = self.coef[0] + Xp.to_numpy() @ self.coef[1:]
        floor = np.nanpercentile(rv, 1)                  # keep variance forecasts positive
        return forecasts.clip(lower=floor)
