"""Section 6 of the original func_lib, pricing part: Black-Scholes, Greeks, implied vol,
GBM paths, Monte Carlo pricing and the GBM class.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._common import TRADING_DAYS


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
