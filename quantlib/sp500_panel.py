"""Cross-sectional research tools for the revisited regression project.

Everything works on WIDE frames (dates x tickers). Plain functions, no hidden state; this module
is written in the shape `func_lib.py`'s cross-sectional section will take when it is split out.

    load_sp500_panel      frozen prices and index membership by date (point-in-time or not)
    momentum_features     past returns over several windows
    forward_return        the return over the next h days (the thing being predicted)
    sector_rank           each value's rank inside its sector on that date, centred on zero
    stack                 wide frames -> one long (date, ticker) table for model fitting
    unstack_scores        model output -> a wide score frame
    daily_rank_ic         per-date rank correlation of score and outcome, sector-neutral
    ic_summary            mean IC and a t-statistic that allows for overlapping horizons
    long_flat_returns     the original rule: hold each stock when its score is positive, else cash
    long_short_returns    sector-neutral top-fifth minus bottom-fifth
    perf                  annual return, volatility, Sharpe ratio, worst drawdown
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'
TRADING_DAYS = 252


@dataclass
class Panel:
    close: pd.DataFrame          # dates x tickers, adjusted close
    sectors: pd.Series           # ticker -> GICS sector
    member: pd.DataFrame         # bool, dates x tickers: counted as in the universe on that date


def load_sp500_panel(point_in_time: bool = True) -> Panel:
    """Frozen S&P 500 snapshot, 2006 to the end of 2021 (later data is deliberately not in the file).

    point_in_time=True   a company is in the universe only from the date it joined the index.
    point_in_time=False  every current member for its whole price history: the original project's
                         practice, which lets the future (who ends up in the index) leak into the past.
    Companies that LEFT the index are missing either way (free data has no removed members).
    """
    close = pd.read_csv(DATA / 'sp500_pit_close_2006_2021.csv.gz', index_col=0, parse_dates=True).sort_index()
    meta = pd.read_csv(DATA / 'sp500_pit_constituents.csv', parse_dates=['date_added']).set_index('symbol').reindex(close.columns)
    member = close.notna()
    if point_in_time:
        joined = meta['date_added'].fillna(pd.Timestamp('1900-01-01'))
        member &= pd.DataFrame(close.index.to_numpy()[:, None] >= joined.to_numpy()[None, :], index=close.index, columns=close.columns)
    return Panel(close, meta['sector'], member)


def momentum_features(close: pd.DataFrame, windows: Iterable[int]) -> Dict[str, pd.DataFrame]:
    """{f'{m}_d_returns': the return over the past m days}, known at each day's close."""
    return {f'{m}_d_returns': close.pct_change(m, fill_method=None) for m in windows}


def forward_return(close: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Return from today's close to the close h days later (NaN for the last h days)."""
    return close.pct_change(h, fill_method=None).shift(-h)


def sector_rank(x: pd.DataFrame, sectors: pd.Series, member: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rank of each value among its sector's members on that date, mapped to (-0.5, 0.5).
    Removes the sector's own move, bounds outliers, and gives every feature the same scale."""
    x = x.where(member) if member is not None else x
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for _, cols in sectors.dropna().groupby(sectors.dropna()).groups.items():
        cols = [c for c in cols if c in x.columns]
        block = x[cols]
        n = block.notna().sum(axis=1)
        out[cols] = (block.rank(axis=1) - 0.5).div(n.where(n >= 5), axis=0) - 0.5      # sectors with < 5 names: no rank
    return out


def stack(frames: Dict[str, pd.DataFrame], member: pd.DataFrame, dates=None, every: int = 1) -> pd.DataFrame:
    """Long table indexed by (Date, Ticker) with one column per frame, members only, complete rows only.
    `every` keeps every n-th date (overlapping long-horizon targets add little information)."""
    idx = member.index if dates is None else member.index[(member.index >= dates[0]) & (member.index <= dates[1])]
    idx = idx[::every]
    cols = {k: v.reindex(idx).where(member.reindex(idx)).stack(future_stack=True) for k, v in frames.items()}
    out = pd.concat(cols, axis=1).dropna(how='any')
    out.index.names = ['Date', 'Ticker']
    return out


def unstack_scores(scores, index: pd.MultiIndex, like: pd.DataFrame) -> pd.DataFrame:
    """Model predictions for the rows of `index` -> a wide dates x tickers frame shaped like `like`."""
    s = pd.Series(np.asarray(scores).ravel(), index=index)
    return s.unstack('Ticker').reindex(index=like.index, columns=like.columns)


def daily_rank_ic(score: pd.DataFrame, outcome: pd.DataFrame, sectors: pd.Series, member: pd.DataFrame) -> pd.Series:
    """Per date: the correlation between the within-sector rank of the score and the within-sector
    rank of the outcome. 'Does the model order stocks correctly against their own sector?'"""
    a, b = sector_rank(score, sectors, member), sector_rank(outcome, sectors, member)
    ok = a.notna() & b.notna()
    a, b = a.where(ok), b.where(ok)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return (num / den.where(den > 0)).where(ok.sum(axis=1) >= 50)


def ic_summary(ic: pd.Series, h: int = 1) -> dict:
    """Mean IC and its t-statistic. With an h-day horizon consecutive daily ICs share h - 1 days of
    returns, so the number of independent observations is taken as n / h."""
    ic = ic.dropna()
    n_eff = max(len(ic) / h, 2)
    return {'IC': float(ic.mean()), 't': float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n_eff))), 'days': int(len(ic))}


def long_flat_returns(score: pd.DataFrame, fwd1: pd.DataFrame, member: pd.DataFrame, threshold: float = 0.0,
                      cost_bps: float = 0.0) -> pd.DataFrame:
    """The original rule. Each member stock is held (1) when its score beats the threshold, else cash (0);
    the daily return is the average over ALL member stocks. Cost: cost_bps on every dollar traded,
    i.e. on the average absolute change in position. Returns columns ret, turnover, time_long."""
    valid = member & score.notna() & fwd1.notna()
    pos = ((score > threshold) & valid).astype(float)
    n = valid.sum(axis=1).where(lambda s: s > 0)
    gross = (pos * fwd1.fillna(0.0)).sum(axis=1) / n
    turnover = pos.diff().abs().sum(axis=1) / n
    turnover.iloc[0] = pos.iloc[0].sum() / n.iloc[0]
    return pd.DataFrame({'ret': gross - cost_bps / 1e4 * turnover, 'turnover': turnover, 'time_long': pos.sum(axis=1) / n}).dropna()


def benchmark_returns(fwd1: pd.DataFrame, member: pd.DataFrame) -> pd.Series:
    """Equal-weighted average of every member stock, every day (always fully invested)."""
    return fwd1.where(member).mean(axis=1).dropna()


def long_short_returns(score: pd.DataFrame, fwd1: pd.DataFrame, sectors: pd.Series, member: pd.DataFrame,
                       quantile: float = 0.2, every: int = 1, cost_bps: float = 0.0) -> pd.DataFrame:
    """Sector-neutral long-short: within each sector, long the top `quantile` of scores and short the
    bottom, equal weight, $1 long and $1 short in total. Weights are reset every `every` days and
    held in between. Cost: cost_bps on every dollar traded. Returns columns ret, turnover."""
    r = sector_rank(score, sectors, member & fwd1.notna())
    long, short = (r > 0.5 - quantile).astype(float), (r < -0.5 + quantile).astype(float)
    w = long.div(long.sum(axis=1).where(lambda s: s > 0), axis=0) - short.div(short.sum(axis=1).where(lambda s: s > 0), axis=0)
    w = w.dropna(how='all').fillna(0.0)
    if every > 1:
        keep = np.arange(len(w)) % every == 0
        w = w.where(pd.Series(keep, index=w.index), axis=0).ffill()
    gross = (w * fwd1.reindex(w.index).fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    return pd.DataFrame({'ret': gross - cost_bps / 1e4 * turnover, 'turnover': turnover})


def perf(r: pd.Series) -> dict:
    """Annualized return, volatility and Sharpe ratio (no risk-free rate), growth of $1, worst drawdown."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {'ann_return': float(r.mean() * TRADING_DAYS), 'ann_vol': float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            'sharpe': float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan,
            'growth_of_1': float(eq.iloc[-1]), 'max_drawdown': float((1 - eq / eq.cummax()).max())}


def active_t(strategy: pd.Series, benchmark: pd.Series) -> float:
    """t-statistic of the daily return difference, strategy minus benchmark."""
    d = (strategy - benchmark).dropna()
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
