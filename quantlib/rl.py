"""Section 5 of the original func_lib: reinforcement-learning (DQN) trading agent.
"""

import random
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import expit

from .theme import DOWN, NEUTRAL, PALETTE, RANGE_SELECTOR, UP, _rgba, subtitled


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


class EpisodeResult(dict):
    """What run_episode returns: a plain dict of the episode's outputs, plus attribute access.

    It is a dict subclass, so everything that worked on the old result dict still works:
    result['pnl'], result.get('pnl'), 'pnl' in result, iteration, .keys() / .items() / .values(),
    dict(result), isinstance(result, dict), pickling. On top of that, result.pnl is result['pnl'],
    and a few convenience methods call the existing report functions.

    Keys from run_episode: buys, sells, trades (DataFrame), losses, actions, pnl, inventory,
    open_positions. Attribute access is read-only; assign through result['key'] = value.
    """
    __slots__ = ()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'{type(self).__name__} has no field {name!r}') from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | {k for k in self if isinstance(k, str) and k.isidentifier()})

    def summary(self, title=''):
        """Print the episode summary (print_summary)."""
        print_summary(self, title)

    def trade_report(self, df):
        """Closed trades with timestamps and market context from `df` (trade_report)."""
        return trade_report(df, self)


class TradingEnv:
    """The single-stock trading environment that run_episode steps through, with reset() / step().

    One share is bought per buy action (up to `max_inventory` lots) and lots are sold FIFO.
    The state at t is the market state states[t] plus position_features(), so the agent knows what
    it holds. Rewards follow `reward_mode` ('mtm' daily mark-to-market in % of price, net of costs,
    or 'realized' = max(trade profit, 0) on sells), exactly as described in run_episode.

    Usage
        env = TradingEnv(states, prices, max_inventory=5, reward_mode='mtm', cost_bps=5)
        state = env.reset()
        while not env.done:
            action = my_policy(state, env.valid_actions())
            state, reward, done, info = env.step(action)
        result = env.finish()            # EpisodeResult, same keys as run_episode

    step() does not enforce the mask: it does what it is told, as run_episode always has (a sell
    with nothing held raises IndexError). Use valid_actions() to choose an allowed action.
    """

    def __init__(self, states, prices, max_inventory=5, reward_mode='mtm', cost_bps=0.0, verbose=False):
        self.states, self.prices = states, prices
        self.max_inventory, self.reward_mode, self.cost_bps = max_inventory, reward_mode, cost_bps
        self.verbose = verbose
        self.n = len(states)
        self.last_t = self.n - 1
        self.inventory = deque()          # FIFO open lots: (buy_price, buy_index); cleared, never replaced
        self.t = None

    def reset(self):
        """Empty book at t = 0; returns the first state."""
        n = self.n
        self.inventory.clear()
        self.buys, self.sells, self.trades = [], [], []
        self.actions = np.zeros(n, dtype=int)
        self.pnl = np.zeros(n)          # cumulative realized + unrealized P&L, $ per share, net of costs
        self.held = np.zeros(n, dtype=int)
        self.realized = 0.0
        self.t = 0
        self.state = self.full_state(0)
        return self.state

    @property
    def done(self):
        """True once the last state has been reached (no more steps)."""
        return self.t is not None and self.t >= self.last_t

    def full_state(self, t):
        """Market state at t plus the position features of the current inventory."""
        return np.concatenate([self.states[t],
                               position_features(self.inventory, self.prices[t], self.max_inventory)])

    def valid_actions(self):
        """Action mask: always hold; buy unless the inventory is full; sell only if something is held."""
        inventory = self.inventory
        return [0] + ([1] if len(inventory) < self.max_inventory else []) + ([2] if inventory else [])

    def _sell(self, t, reason):
        prices = self.prices
        buy_price, buy_t = self.inventory.popleft()
        profit = prices[t] - buy_price
        cost = prices[t] * self.cost_bps / 1e4
        self.realized += profit - cost
        self.sells.append(t)
        self.trades.append({'buy_t': buy_t, 'sell_t': t, 'Buy Price': buy_price, 'Sell Price': prices[t],
                            'Trade Profit': profit, 'Exit Reason': reason})
        if self.verbose:
            print(f'Sell: {format_price(prices[t])} | Profit: {format_price(profit)}')
        return profit

    def step(self, action):
        """Apply `action` (0 hold, 1 buy, 2 sell oldest) at t, then mark the book to t + 1.

        Returns (next_state, reward, done, info); info has 't' (the step just taken) and
        'realized_reward' (max(trade profit, 0) of a sell, else 0).
        """
        if self.t is None:
            raise RuntimeError('call reset() before step()')
        if self.done:
            raise RuntimeError('the episode is over; call finish() or reset()')
        t, prices, inventory = self.t, self.prices, self.inventory
        self.actions[t] = action
        price = prices[t]
        realized_reward = 0.0

        if action == 1:                                   # buy 1 share
            inventory.append((price, t))
            self.realized -= price * self.cost_bps / 1e4
            self.buys.append(t)
            if self.verbose:
                print(f'Buy:  {format_price(price)}')
        elif action == 2:                                 # sell the oldest lot
            realized_reward = max(self._sell(t, 'Agent sell'), 0.0)

        # Mark the book to t+1
        pnl = self.pnl
        unrealized_next = sum(prices[t + 1] - p for p, _ in inventory)
        pnl[t + 1] = self.realized + unrealized_next
        self.held[t + 1] = len(inventory)
        if self.reward_mode == 'mtm':
            reward = 100 * (pnl[t + 1] - pnl[t]) / price
        else:
            reward = realized_reward

        next_state = self.full_state(t + 1)
        done = t == self.last_t - 1
        self.t, self.state = t + 1, next_state
        return next_state, reward, done, {'t': t, 'realized_reward': realized_reward}

    def finish(self, losses=None, close_at_end=True):
        """End the episode and return its EpisodeResult (the same keys and values run_episode returns).

        close_at_end: sell any lots still open on the last day ('End of data'), so every buy shows up
        in the trade log. `losses` is the list of replay losses to store in the result (default []).
        Call it once per episode.
        """
        inventory, last_t = self.inventory, self.last_t
        open_positions = len(inventory)
        if close_at_end:
            while inventory:
                self._sell(last_t, 'End of data')
            self.pnl[last_t] = self.realized
            self.held[last_t] = 0

        trades = pd.DataFrame(self.trades, columns=['buy_t', 'sell_t', 'Buy Price', 'Sell Price',
                                                    'Trade Profit', 'Exit Reason'])
        trades = trades.astype({'buy_t': int, 'sell_t': int, 'Trade Profit': float})   # also when empty
        return EpisodeResult({'buys': self.buys, 'sells': self.sells, 'trades': trades,
                              'losses': [] if losses is None else losses, 'actions': self.actions,
                              'pnl': self.pnl, 'inventory': self.held, 'open_positions': open_positions})


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

    Returns an EpisodeResult (a dict) with buys, sells, trades (DataFrame), losses, actions, the
    cumulative P&L path in $ per share ('pnl'), the inventory path, and open_positions (lots left
    before the final close).

    The simulation itself lives in TradingEnv; this function is the training / evaluation loop over it.
    """
    env = TradingEnv(states, prices, max_inventory, reward_mode, cost_bps, verbose)
    if agent is not None:
        agent.inventory = env.inventory
    losses = []
    state = env.reset()
    for t in progress(range(env.last_t), desc=desc):
        valid = env.valid_actions()
        action = policy(t, state, valid) if policy is not None else agent.act(state, valid)
        next_state, reward, done, _ = env.step(action)
        if train and agent is not None:
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size and t % train_every == 0:
                losses.append(agent.exp_replay(batch_size))
        state = next_state
    return env.finish(losses, close_at_end)


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


# ---- Generic tabular Q-learning loop (moved here from the clustering section; no notebook calls it) ----
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
