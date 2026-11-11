"""Section 7 of the original func_lib: RL options trading agent (buy / sell calls and
puts, close, hold) with trade tracking.
"""

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from ._common import TRADING_DAYS
from .theme import DOWN, NEUTRAL, PALETTE, UP, _rgba, subtitled
from .rl import EpisodeResult, format_price, progress
from .options import bs_greeks, bs_price


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


class OptionsEpisodeResult(EpisodeResult):
    """What run_options_episode returns: a dict of the episode's outputs, plus attribute access.

    Keys: equity (Series), trades (DataFrame), losses, actions, entries, exits, counts, margin
    (Series), open_positions. See EpisodeResult for the dict compatibility.
    """
    __slots__ = ()

    def summary(self, title=''):
        """Print the options episode summary (print_options_summary)."""
        print_options_summary(self, title)

    def trade_statistics(self, groups=('Side', 'Type', 'Position')):
        """Trade statistics for All trades and per side, type and position (options_trade_statistics)."""
        return options_trade_statistics(self['trades'], groups)

    def trade_report(self, df=None):
        """A copy of the trade log. Options trades already carry their dates and entry context,
        so `df` is not needed and is ignored."""
        return self['trades'].copy()


class OptionsEnv:
    """The options-book environment that run_options_episode steps through, with reset() / step().

    market_states, md and cfg are exactly as in run_options_episode, which documents the trading
    rules: 1-contract European calls / puts struck at S*(1 +/- moneyness), longs pay the ask and
    shorts collect the bid, shorts post Reg-T style margin and are force-closed ('Margin call') when
    equity falls below margin used, positions are marked at BS(iv) and settle at intrinsic on expiry.
    An action that can't be done becomes a hold. Reward = daily change in account equity as a percent
    of starting capital.

    State held here: the open book (one dict per position, in opening order), cash, the equity and
    margin paths, per-position-type counts, entries / exits and the trade log.

    Usage
        env = OptionsEnv(market_states, md, cfg)
        state = env.reset()
        while not env.done:
            state, reward, done, info = env.step(my_policy(env.t, state))
        result = env.finish()            # OptionsEpisodeResult, same keys as run_options_episode
    """

    def __init__(self, market_states, md, cfg, verbose=False):
        self.market_states, self.md, self.cfg = market_states, md, cfg
        self.S, self.iv, self.rf, self.dates = md['S'], md['iv'], md['rf'], md['dates']
        self.short_allowed = md.get('short_allowed')
        self.tenor, self.mult, self.cap = cfg['tenor_days'], cfg['contract_mult'], cfg['capital']
        self.max_pos = cfg['max_positions']
        self.max_short = cfg.get('max_short_positions', self.max_pos)
        self.cost, self.m = cfg['cost_vol'], cfg.get('moneyness', 0.0)
        self.m_pct, self.m_min = cfg.get('short_margin_pct', 0.20), cfg.get('min_margin_pct', 0.10)
        self.verbose = verbose
        self.n = len(self.S)
        self.t = None

    def reset(self):
        """Empty book and full capital at t = 0; returns the first state."""
        n, cap = self.n, self.cap
        self.book, self.trades, self.exits = [], [], []
        self.actions = np.zeros(n, dtype=int)
        self.entries = {name: [] for name in POSITION_ORDER}
        self.counts = {name: np.zeros(n, dtype=int) for name in POSITION_ORDER}
        self.cash = float(cap)
        self.equity, self.margin = np.full(n, np.nan), np.zeros(n)
        self.equity[0] = cap
        self.t = 0
        self.state = np.concatenate([self.market_states[0],
                                     _book_features(self.book, 0, self.max_pos, self.tenor, 0.0, cap)])
        return self.state

    @property
    def done(self):
        """True once the last day has been reached (no more steps)."""
        return self.t is not None and self.t >= self.n - 1

    def value(self, p, t, vol_shift=0.0):
        """Value of position `p` at t: BS(iv + vol_shift) before expiry, intrinsic value at expiry."""
        S, iv, rf = self.S, self.iv, self.rf
        tau = max(p['expiry_t'] - t, 0) / TRADING_DAYS
        kind = p['Type'].lower()
        if tau == 0:
            return max(S[t] - p['Strike'], 0) if kind == 'call' else max(p['Strike'] - S[t], 0)
        return float(bs_price(S[t], p['Strike'], tau, rf[t], max(iv[t] + vol_shift, 0.01), 0.0, kind))

    def margin_used(self, t):
        """Total margin required at t by the short positions in the book."""
        S, mult, m_pct, m_min = self.S, self.mult, self.m_pct, self.m_min
        return sum(short_margin_requirement(p['Type'].lower(), S[t], p['Strike'], self.value(p, t), mult, m_pct, m_min)
                   for p in self.book if p['sign'] < 0)

    def mark(self, t):
        """Account equity at t: cash plus the signed value of every open position."""
        return self.cash + sum(p['sign'] * self.value(p, t) * self.mult for p in self.book)

    def _close(self, p, t, reason, use_spread=True):
        # Longs sell at the bid, shorts buy back at the ask; expiry and end-of-data use intrinsic / mid
        S, dates, mult = self.S, self.dates, self.mult
        exit_value = self.value(p, t, -p['sign'] * self.cost if use_spread else 0.0)
        sign = p['sign']
        pnl = sign * (exit_value - p['Entry Premium']) * mult
        ret = 100 * sign * (exit_value - p['Entry Premium']) / p['Entry Premium']
        self.trades.append({**{k: v for k, v in p.items() if k not in ('entry_t', 'expiry_t', 'sign')},
                            'Exit Date': dates[t], 'Exit Reason': reason, 'Exit Spot': S[t], 'Exit Value': exit_value,
                            'P&L ($)': pnl, 'Return %': ret, 'Holding Days': t - p['entry_t']})
        self.exits.append(t)
        if self.verbose:
            print(f"{dates[t]:%Y-%m-%d} close {p['Position']} K={p['Strike']:.0f} ({reason}) P&L {format_price(pnl)}")
        return sign * exit_value * mult           # cash flow: +proceeds for longs, -buy-back cost for shorts

    def step(self, action):
        """Apply `action` (OPTION_ACTIONS) at t, then move to t + 1: expiries, mark-to-market, margin calls.

        Returns (next_state, reward, done, info); info has 't', the step just taken.
        """
        if self.t is None:
            raise RuntimeError('call reset() before step()')
        if self.done:
            raise RuntimeError('the episode is over; call finish() or reset()')
        t, n, book = self.t, self.n, self.book
        S, iv, rf, dates, md = self.S, self.iv, self.rf, self.dates, self.md
        tenor, mult, cost, m = self.tenor, self.mult, self.cost, self.m
        self.actions[t] = action

        if action in _OPEN_ACTIONS and len(book) < self.max_pos:
            kind, sign = _OPEN_ACTIONS[action]
            strike = S[t] * (1 + m) if kind == 'call' else S[t] * (1 - m)
            T = tenor / TRADING_DAYS
            premium = float(bs_price(S[t], strike, T, rf[t], max(iv[t] + sign * cost, 0.01), 0.0, kind))
            free_equity = self.mark(t) - self.margin_used(t)
            if sign > 0:
                ok = free_equity >= premium * mult
            else:
                n_short = sum(p['sign'] < 0 for p in book)
                allowed = True if self.short_allowed is None else bool(self.short_allowed[t])
                new_margin = short_margin_requirement(kind, S[t], strike, premium, mult, self.m_pct, self.m_min)
                ok = allowed and n_short < self.max_short and free_equity >= new_margin
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
                self.cash -= sign * premium * mult      # pay for longs, collect for shorts
                self.entries[p['Position']].append(t)
                if self.verbose:
                    print(f"{dates[t]:%Y-%m-%d} {side.lower()} {kind} K={strike:.0f} @ {format_price(premium)}")
        elif action == 5 and book:
            self.cash += self._close(book.pop(0), t, 'Agent close')

        # ---- move to t+1: expiries, mark-to-market, margin calls ----
        t1 = t + 1
        for p in [p for p in book if p['expiry_t'] <= t1]:
            book.remove(p)
            self.cash += self._close(p, t1, 'Expired', use_spread=False)
        while any(p['sign'] < 0 for p in book) and self.mark(t1) < self.margin_used(t1):
            oldest_short = next(p for p in book if p['sign'] < 0)
            book.remove(oldest_short)
            self.cash += self._close(oldest_short, t1, 'Margin call')

        equity, margin = self.equity, self.margin
        equity[t1], margin[t1] = self.mark(t1), self.margin_used(t1)
        for name in POSITION_ORDER:
            self.counts[name][t1] = sum(p['Position'] == name for p in book)

        reward = 100 * (equity[t1] - equity[t]) / self.cap
        next_state = np.concatenate([self.market_states[t1],
                                     _book_features(book, t1, self.max_pos, tenor, margin[t1], equity[t1])])
        done = t1 == n - 1
        self.t, self.state = t1, next_state
        return next_state, reward, done, {'t': t}

    def finish(self, losses=None):
        """End the episode: close what is left at the last mark (mid, 'End of data') and return the
        OptionsEpisodeResult (the same keys and values run_options_episode returns).

        `losses` is the list of replay losses to store in the result (default []). Call it once per
        episode. As before, the end-of-data closes are logged as trades but not added to cash.
        """
        n, book, dates = self.n, self.book, self.dates
        open_positions = len(book)
        for p in list(book):                                     # close what is left at the last mark (mid)
            book.remove(p)
            self._close(p, n - 1, 'End of data', use_spread=False)

        return OptionsEpisodeResult({
            'equity': pd.Series(self.equity, index=dates, name='Equity'), 'trades': pd.DataFrame(self.trades),
            'losses': [] if losses is None else losses, 'actions': self.actions, 'entries': self.entries,
            'exits': self.exits, 'counts': {k: pd.Series(v, index=dates) for k, v in self.counts.items()},
            'margin': pd.Series(self.margin, index=dates, name='Margin used'), 'open_positions': open_positions})


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

    Returns an OptionsEpisodeResult (a dict). The simulation itself lives in OptionsEnv; this function
    is the training / evaluation loop over it.
    """
    env = OptionsEnv(market_states, md, cfg, verbose=verbose)
    losses = []
    state = env.reset()
    for t in progress(range(env.n - 1), desc=desc):
        action = policy(t, state) if policy is not None else agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if train and agent is not None:
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size and t % train_every == 0:
                losses.append(agent.exp_replay(batch_size))
        state = next_state
    return env.finish(losses)


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
