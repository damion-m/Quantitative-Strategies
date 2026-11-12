"""Tests for the step-2 classes in quantlib and for the functions that now wrap them.

Run from the repository root:
    python -m unittest discover -s tests

No network access is needed. The Keras tests are skipped when keras is not installed and take a
few seconds. Charts are built but never shown (Figure.show is patched out).
"""
import copy
import io
import os
import pickle
import random
import sys
import unittest
from collections import deque
from contextlib import redirect_stdout
from unittest import mock

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')   # keep TensorFlow's C++ log noise out of the test output

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # the repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import plotly.graph_objects as go       # noqa: E402

from quantlib import func_lib           # noqa: E402
import quantlib                         # noqa: E402
from quantlib.func_lib import (         # noqa: E402
    ACTIONS, OPTION_ACTIONS, EpisodeResult, GarchForecaster, HarForecaster, OptionContract,
    OptionsEnv, OptionsEpisodeResult, TradingEnv, run_episode, run_options_episode,
)

try:
    import keras                        # noqa: F401
    HAVE_KERAS = True
except ImportError:                     # pragma: no cover
    HAVE_KERAS = False

OPTION_CFG = dict(tenor_days=21, moneyness=0.0, cost_vol=0.01, contract_mult=100, capital=100_000,
                  max_positions=6, max_short_positions=3, short_margin_pct=0.20, min_margin_pct=0.10)


def quiet(fn, *args, **kwargs):
    """Call fn with stdout captured and stderr (the progress line) silenced."""
    with redirect_stdout(io.StringIO()), mock.patch.object(sys, 'stderr', io.StringIO()):
        return fn(*args, **kwargs)


def stock_data(n=120, seed=3):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    states = func_lib.build_states(rng.normal(size=(n, 7)).cumsum(axis=0), 1)
    dates = pd.bdate_range('2022-01-03', periods=n)
    df = pd.DataFrame({'Close': prices, 'SMA_20': pd.Series(prices).rolling(5, min_periods=1).mean().to_numpy(),
                       'Volatility': rng.random(n)}, index=dates)
    df['BB_upper'], df['BB_lower'] = df['Close'] * 1.05, df['Close'] * 0.95
    return states, prices, df


def options_data(n=140, seed=5, crash=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2021-01-04', periods=n)
    r = rng.normal(0.0003, 0.012, n)
    if crash:
        r[60:75] = -0.035
    S = 400 * np.exp(np.cumsum(r))
    iv = np.clip(0.18 + 0.05 * np.sin(np.arange(n) / 9) + rng.normal(0, 0.01, n), 0.08, None)
    if crash:
        iv[60:80] += 0.25
    md = {'dates': dates, 'S': S, 'iv': iv, 'rf': np.full(n, 0.03), 'sigma_f': iv * 0.9,
          'mu': np.full(n, 0.06), 'call_edge': rng.normal(0, 0.1, n), 'put_edge': rng.normal(0, 0.1, n),
          'short_allowed': rng.random(n) > 0.15}
    z = np.column_stack([S, iv, md['call_edge'], md['put_edge']])
    z = (z - z.mean(0)) / z.std(0)
    return func_lib.build_states(z, 1), md


def assert_same(test, a, b, path='result'):
    """Exact equality for the objects an episode result holds."""
    test.assertIs(type(a), type(b), path)
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_exact=True)
    elif isinstance(a, np.ndarray):
        test.assertEqual(a.dtype, b.dtype, path)
        test.assertTrue(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), path)
    elif isinstance(a, dict):
        test.assertEqual(list(a), list(b), path)
        for k in a:
            assert_same(test, a[k], b[k], f'{path}[{k!r}]')
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f'{path}[{i}]')
    else:
        test.assertEqual(a, b, path)


class StubAgent:
    """Deterministic agent with the interface run_episode / run_options_episode use."""

    def __init__(self, n_actions=3, epsilon=0.6):
        self.memory, self.inventory = deque(maxlen=200), deque()
        self.epsilon, self.n_actions = epsilon, n_actions

    def act(self, state, valid_actions=None):
        actions = list(range(self.n_actions)) if valid_actions is None else list(valid_actions)
        if random.random() <= self.epsilon:
            return random.choice(actions)
        return actions[int(abs(float(np.sum(state))) * 1000) % len(actions)]

    def remember(self, *transition):
        self.memory.append(transition)

    def exp_replay(self, batch_size=32):
        return float(sum(t[2] for t in random.sample(self.memory, batch_size)))


# ---------------------------------------------------------------------------------------------
class TestEpisodeResult(unittest.TestCase):
    def setUp(self):
        states, prices, _ = stock_data()
        rng = np.random.default_rng(0)
        self.res = quiet(run_episode, None, states, prices, policy=lambda t, s, v: int(rng.choice(v)), cost_bps=5)

    def test_is_a_dict(self):
        r = self.res
        self.assertIsInstance(r, dict)
        self.assertIsInstance(r, EpisodeResult)
        keys = ['buys', 'sells', 'trades', 'losses', 'actions', 'pnl', 'inventory', 'open_positions']
        self.assertEqual(list(r), keys)
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(len(r.items()), len(keys))
        self.assertEqual(len(list(r.values())), len(keys))
        self.assertIn('pnl', r)
        self.assertNotIn('equity', r)
        self.assertIs(r.get('pnl'), r['pnl'])
        self.assertIsNone(r.get('missing'))
        plain = dict(r)
        self.assertIs(type(plain), dict)
        self.assertEqual(list(plain), keys)
        self.assertIs(plain['trades'], r['trades'])
        self.assertEqual(dict(**r).keys(), r.keys())

    def test_attribute_access(self):
        r = self.res
        self.assertIs(r.pnl, r['pnl'])
        self.assertIs(r.trades, r['trades'])
        with self.assertRaises(AttributeError):
            r.not_a_field
        with self.assertRaises(AttributeError):
            r.pnl = 1                   # read-only: write through r['pnl']
        self.assertFalse(hasattr(r, '__deepcopy_missing__'))
        self.assertIn('pnl', dir(r))
        r['extra'] = 1                  # still a normal, mutable dict
        self.assertEqual(r.extra, 1)

    def test_pickle_and_copy(self):
        r = self.res
        for clone in (pickle.loads(pickle.dumps(r)), copy.deepcopy(r), copy.copy(r)):
            self.assertIs(type(clone), EpisodeResult)
            assert_same(self, dict(r), dict(clone))

    def test_convenience_methods(self):
        r = self.res
        _, _, df = stock_data()
        out = io.StringIO()
        with redirect_stdout(out):
            r.summary('t')
        ref = io.StringIO()
        with redirect_stdout(ref):
            func_lib.print_summary(r, 't')
        self.assertEqual(out.getvalue(), ref.getvalue())
        pd.testing.assert_frame_equal(r.trade_report(df), func_lib.trade_report(df, r))

    def test_options_result(self):
        states, md = options_data()
        rng = np.random.default_rng(1)
        r = quiet(run_options_episode, states, md, OPTION_CFG, policy=lambda t, s: int(rng.integers(6)))
        self.assertIsInstance(r, dict)
        self.assertIsInstance(r, OptionsEpisodeResult)
        self.assertIsInstance(r, EpisodeResult)
        self.assertEqual(list(r), ['equity', 'trades', 'losses', 'actions', 'entries', 'exits', 'counts',
                                   'margin', 'open_positions'])
        self.assertIs(r.equity, r['equity'])
        pd.testing.assert_frame_equal(r.trade_statistics(), func_lib.options_trade_statistics(r['trades']))
        out, ref = io.StringIO(), io.StringIO()
        with redirect_stdout(out):
            r.summary('x')
        with redirect_stdout(ref):
            func_lib.print_options_summary(r, 'x')
        self.assertEqual(out.getvalue(), ref.getvalue())
        clone = pickle.loads(pickle.dumps(r))
        self.assertIs(type(clone), OptionsEpisodeResult)
        assert_same(self, dict(r), dict(clone))


# ---------------------------------------------------------------------------------------------
class TestTradingEnv(unittest.TestCase):
    def test_manual_loop_matches_run_episode(self):
        states, prices, _ = stock_data()
        for reward_mode in ('mtm', 'realized'):
            for close_at_end in (True, False):
                rng = np.random.default_rng(4)
                policy = lambda t, s, v: int(rng.choice(v))          # noqa: E731
                ref = quiet(run_episode, None, states, prices, policy=policy, max_inventory=3,
                            reward_mode=reward_mode, cost_bps=5, close_at_end=close_at_end)

                rng = np.random.default_rng(4)
                env = TradingEnv(states, prices, max_inventory=3, reward_mode=reward_mode, cost_bps=5)
                state, rewards = env.reset(), []
                while not env.done:
                    state, reward, done, info = env.step(int(rng.choice(env.valid_actions())))
                    rewards.append(reward)
                self.assertTrue(done)
                assert_same(self, dict(ref), dict(env.finish(close_at_end=close_at_end)))
                self.assertEqual(len(rewards), len(states) - 1)

    def test_stub_agent_train_matches_manual_loop(self):
        states, prices, _ = stock_data()
        random.seed(9)
        agent = StubAgent()
        ref = quiet(run_episode, agent, states, prices, train=True, batch_size=8, train_every=2, cost_bps=5)
        self.assertIs(agent.inventory.__class__, deque)
        self.assertGreater(len(ref['losses']), 0)

        random.seed(9)
        agent2 = StubAgent()
        env = TradingEnv(states, prices, cost_bps=5)
        agent2.inventory = env.inventory
        state, losses = env.reset(), []
        for t in range(env.last_t):
            action = agent2.act(state, env.valid_actions())
            next_state, reward, done, _ = env.step(action)
            agent2.remember(state, action, reward, next_state, done)
            if len(agent2.memory) > 8 and t % 2 == 0:
                losses.append(agent2.exp_replay(8))
            state = next_state
        assert_same(self, dict(ref), dict(env.finish(losses)))

    def test_mask_state_and_guards(self):
        states, prices, _ = stock_data(n=10)
        env = TradingEnv(states, prices, max_inventory=2)
        with self.assertRaises(RuntimeError):
            env.step(0)
        state = env.reset()
        self.assertEqual(state.shape, (states.shape[1] + func_lib.N_POSITION_FEATURES,))
        self.assertEqual(env.valid_actions(), [0, 1])
        env.step(1)
        env.step(1)
        self.assertEqual(env.valid_actions(), [0, 2])
        self.assertEqual(len(env.inventory), 2)
        while not env.done:
            env.step(0)
        with self.assertRaises(RuntimeError):
            env.step(0)
        res = env.finish()
        self.assertEqual(res['open_positions'], 2)
        self.assertEqual(list(res['trades']['Exit Reason']), ['End of data', 'End of data'])
        self.assertEqual(res['pnl'][-1], env.realized)

    def test_agent_inventory_side_effect(self):
        states, prices, _ = stock_data(n=30)
        agent = StubAgent(epsilon=0.0)
        quiet(run_episode, agent, states, prices, close_at_end=False)
        self.assertIsInstance(agent.inventory, deque)       # run_episode still hands the agent its inventory


class TestOptionsEnv(unittest.TestCase):
    def test_manual_loop_matches_run_options_episode(self):
        states, md = options_data()
        for capital in (100_000, 20_000):
            cfg = dict(OPTION_CFG, capital=capital)
            rng = np.random.default_rng(2)
            ref = quiet(run_options_episode, states, md, cfg, policy=lambda t, s: int(rng.integers(6)))

            rng = np.random.default_rng(2)
            env = OptionsEnv(states, md, cfg)
            state = env.reset()
            while not env.done:
                state, reward, done, info = env.step(int(rng.integers(6)))
            assert_same(self, dict(ref), dict(env.finish()))

    def test_margin_calls_and_position_caps(self):
        states, md = options_data()
        res = quiet(run_options_episode, states, md, dict(OPTION_CFG, capital=20_000), policy=lambda t, s: 4)
        self.assertGreater((res['trades']['Exit Reason'] == 'Margin call').sum(), 0)
        res = quiet(run_options_episode, states, md, OPTION_CFG, policy=lambda t, s: 1)
        self.assertEqual(int(sum(res['counts'].values()).max()), OPTION_CFG['max_positions'])
        res = quiet(run_options_episode, states, md, OPTION_CFG, policy=lambda t, s: 3)
        shorts = res['counts']['Short Call'] + res['counts']['Short Put']
        self.assertEqual(int(shorts.max()), OPTION_CFG['max_short_positions'])

    def test_stub_agent_train(self):
        states, md = options_data()
        random.seed(5)
        ref = quiet(run_options_episode, states, md, OPTION_CFG, agent=StubAgent(6, 0.7), train=True,
                    batch_size=8, train_every=3)
        random.seed(5)
        agent = StubAgent(6, 0.7)
        env = OptionsEnv(states, md, OPTION_CFG)
        state, losses = env.reset(), []
        for t in range(env.n - 1):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 8 and t % 3 == 0:
                losses.append(agent.exp_replay(8))
            state = next_state
        assert_same(self, dict(ref), dict(env.finish(losses)))

    def test_mark_and_margin_helpers(self):
        states, md = options_data(n=40)
        env = OptionsEnv(states, md, OPTION_CFG)
        env.reset()
        self.assertEqual(env.mark(0), OPTION_CFG['capital'])
        self.assertEqual(env.margin_used(0), 0)
        env.step(4)                                   # sell a put (if allowed that day)
        if env.book:
            p = env.book[0]
            expected = func_lib.short_margin_requirement('put', md['S'][1], p['Strike'], env.value(p, 1))
            self.assertAlmostEqual(env.margin_used(1), expected)


# ---------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_KERAS, 'keras not installed')
class TestKerasAgent(unittest.TestCase):
    def test_run_episode_with_agent_matches_manual_loop(self):
        import keras
        keras.utils.disable_interactive_logging()
        states, prices, _ = stock_data(n=50)
        size = states.shape[1] + func_lib.N_POSITION_FEATURES

        def run(manual):
            keras.utils.set_random_seed(1)
            agent = func_lib.Agent(state_size=size, memory_size=200, epsilon_decay=0.9, target_update_every=10)
            if not manual:
                return quiet(run_episode, agent, states, prices, train=True, batch_size=8), agent
            env = TradingEnv(states, prices)
            agent.inventory = env.inventory
            state, losses = env.reset(), []
            for t in range(env.last_t):
                action = agent.act(state, env.valid_actions())
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > 8:
                    losses.append(agent.exp_replay(8))
                state = next_state
            return env.finish(losses), agent

        (ref, a1), (res, a2) = run(False), run(True)
        assert_same(self, dict(ref), dict(res))
        for w1, w2 in zip(a1.model.get_weights(), a2.model.get_weights()):
            self.assertTrue(np.array_equal(w1, w2))

    def test_options_agent_test_mode(self):
        import keras
        keras.utils.disable_interactive_logging()
        states, md = options_data(n=60)
        keras.utils.set_random_seed(2)
        agent = func_lib.Agent(state_size=states.shape[1] + func_lib.N_BOOK_FEATURES,
                               action_size=len(OPTION_ACTIONS), test_mode=True)
        ref = quiet(run_options_episode, states, md, OPTION_CFG, agent=agent)
        env = OptionsEnv(states, md, OPTION_CFG)
        state = env.reset()
        while not env.done:
            state, *_ = env.step(agent.act(state))
        assert_same(self, dict(ref), dict(env.finish()))


# ---------------------------------------------------------------------------------------------
class TestNotebookUsagePatterns(unittest.TestCase):
    """Each way the notebooks use the result objects, exercised on synthetic results."""

    def setUp(self):
        self.show = mock.patch.object(go.Figure, 'show', lambda self, *a, **k: None)
        self.show.start()

    def tearDown(self):
        self.show.stop()

    def test_rl_trading_agent_notebook(self):
        states, prices, df = stock_data()
        random.seed(0)
        result = quiet(run_episode, StubAgent(), states, prices, train=True, batch_size=8, cost_bps=5,
                       desc='Episode 1/1', max_inventory=5, reward_mode='mtm', verbose=False)
        quiet(func_lib.print_summary, result, 'Episode')
        mean_loss = np.mean(result['losses']) if result['losses'] else np.nan
        log = pd.DataFrame([{'Episode': 1, 'Net P&L ($/share)': result['pnl'][-1],
                             'Closed Trades': len(result['trades']), 'Mean Loss': mean_loss}]).set_index('Episode')
        func_lib.stats_table_figure(log.T, 'x', index_name='Metric', decimals=3)
        func_lib.plot_behavior(df, result, 'x')
        func_lib.plot_losses(result['losses'], 'x')

        rng = np.random.default_rng(42)
        benchmarks = {'Buy & hold (5 shares)': lambda t, s, valid: 1 if 1 in valid else 0,
                      'Random policy': lambda t, s, valid: int(rng.choice(valid))}
        results = {'DQN agent': result}
        for name, policy in benchmarks.items():
            results[name] = quiet(run_episode, None, states, prices, policy=policy, desc=name, cost_bps=5)
        func_lib.policy_comparison_figure(df.index, results, title='x')
        table = func_lib.policy_stats_table(results)
        func_lib.stats_table_figure(table, 'x', index_name='Metric', pct_rows=('Win Rate', 'Days Holding'))
        self.assertEqual(list(table.columns), list(results))

        counts = pd.Series(result['actions'][:-1]).map(ACTIONS).value_counts()
        self.assertEqual(int(counts.sum()), len(states) - 1)
        _ = len(result['trades']) == 0 or counts.get('sell', 0) == 0

        trade_tracker = func_lib.trade_report(df, result)
        trade_tracker['Exit Reason'].value_counts().to_dict()
        trade_tracker.head(10)
        stats = func_lib.trade_statistics(trade_tracker)
        quiet(func_lib.print_trade_statistics, stats)
        func_lib.trade_analysis_figure(trade_tracker, stats['realized_pnl'])

    def test_gbm_options_notebook(self):
        states, md = options_data()
        random.seed(0)
        result = quiet(run_options_episode, states, md, OPTION_CFG, agent=StubAgent(6), train=True,
                       batch_size=8, train_every=1, desc='Episode 1/1', verbose=False)
        quiet(func_lib.print_options_summary, result, 'x')
        func_lib.plot_options_behavior(result, md['S'], 'x')
        func_lib.plot_losses(result['losses'], 'x')

        rng = np.random.default_rng(42)
        rl_test = quiet(run_options_episode, states, md, OPTION_CFG, agent=StubAgent(6, 0.0), desc='Test')
        rule_test = quiet(run_options_episode, states, md, OPTION_CFG, policy=lambda t, s: 0, desc='GBM rule')
        random_test = quiet(run_options_episode, states, md, OPTION_CFG,
                            policy=lambda t, s: int(rng.integers(len(OPTION_ACTIONS))), desc='Random')
        for name, res in [('DQN agent', rl_test), ('GBM-edge rule', rule_test), ('Random policy', random_test)]:
            quiet(func_lib.print_options_summary, res, name)

        cap = OPTION_CFG['capital']
        curves = {'DQN agent': rl_test['equity'], 'GBM-edge rule': rule_test['equity'],
                  'Random policy': random_test['equity'],
                  'buy & hold': cap * pd.Series(md['S'], index=md['dates']) / md['S'][0]}
        for equity in curves.values():
            _ = equity / equity.cummax() - 1
        perf = pd.DataFrame({name: func_lib.performance_stats(eq.pct_change().dropna()) for name, eq in curves.items()})
        func_lib.stats_table_figure(perf, 'x', pct_rows=('Total Return', 'CAGR'))

        trades = random_test['trades']
        for name, res in [('DQN agent', rl_test), ('GBM-edge rule', rule_test)]:
            func_lib.stats_table_figure(func_lib.options_trade_statistics(res['trades']), 'x',
                                        index_name='Statistic', pct_rows=('Win Rate',))
        trade_cols = ['Position', 'Side', 'Type', 'Entry Date', 'Exit Date', 'Exit Reason', 'Strike', 'Entry Spot',
                      'Exit Spot', 'Entry IV', 'Forecast Vol', 'GBM Edge %', 'Entry Premium', 'Exit Value', 'P&L ($)',
                      'Return %', 'Holding Days', 'Delta', 'Gamma', 'Vega', 'Theta']
        trade_log = trades[trade_cols].copy() if len(trades) else pd.DataFrame(columns=trade_cols)
        self.assertEqual(list(trade_log.columns), trade_cols)
        if len(trades):
            func_lib.options_trade_analysis_figure(trades, 'x')
            month_end = random_test['equity'].resample('ME').last()
            month_end.pct_change()
        action_counts = pd.Series(rl_test['actions'][:-1]).map(OPTION_ACTIONS).value_counts()
        self.assertEqual(int(action_counts.sum()), len(states) - 1)


# ---------------------------------------------------------------------------------------------
class TestForecasters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        n = 1200
        idx = pd.bdate_range('2012-01-02', periods=n)
        h, r = 1e-4, np.empty(n)
        for i in range(n):
            r[i] = np.sqrt(h) * rng.standard_normal()
            h = 2e-6 + 0.08 * r[i] ** 2 + 0.9 * h
        cls.log_ret = pd.Series(r, index=idx)
        cls.rv = cls.log_ret ** 2 * np.exp(rng.normal(0, 0.3, n))
        cls.target = func_lib.forward_realized_variance(cls.log_ret, 21)

    def test_garch(self):
        g = GarchForecaster(21).fit(self.log_ret.iloc[:700])
        self.assertEqual(g.params, func_lib.fit_garch11(self.log_ret.iloc[:700]))
        pd.testing.assert_series_equal(g.forecast(self.log_ret),
                                       func_lib.garch_forecast(self.log_ret, g.params, 21), check_exact=True)
        for horizon, freq in ((21, 'YS'), (10, 'QS')):
            pd.testing.assert_series_equal(GarchForecaster(horizon).walk_forward(self.log_ret, '2015-01-01', freq),
                                           func_lib.walk_forward_garch(self.log_ret, '2015-01-01', horizon, freq),
                                           check_exact=True)
        with self.assertRaises(RuntimeError):
            GarchForecaster().forecast(self.log_ret)

    def test_har(self):
        for horizon, freq in ((21, 'YS'), (10, 'QS')):
            pd.testing.assert_series_equal(
                HarForecaster(horizon).walk_forward(self.rv, self.target, '2015-01-01', freq),
                func_lib.walk_forward_har(self.rv, self.target, '2015-01-01', horizon, freq), check_exact=True)
        m = HarForecaster().fit(self.rv.iloc[:500], self.target.iloc[:500])
        self.assertEqual(m.coef.shape, (4,))
        f = m.forecast(self.rv)
        self.assertEqual(f.name, 'HAR')
        self.assertTrue(f.iloc[:21].isna().all() and f.iloc[21:].notna().all())
        floor = float(f.dropna().median())
        self.assertGreaterEqual(m.forecast(self.rv, floor=floor).min(), floor)
        with self.assertRaises(RuntimeError):
            HarForecaster().forecast(self.rv)


class TestOptionContract(unittest.TestCase):
    def test_matches_functions(self):
        S, sigma = np.linspace(80, 120, 7), np.linspace(0.1, 0.4, 7)
        for kind in ('call', 'put', 'straddle'):
            c = OptionContract(kind, 100.0, 0.25)
            self.assertTrue(np.array_equal(c.price(S, 0.03, sigma, 0.01),
                                           func_lib.bs_price(S, 100.0, 0.25, 0.03, sigma, 0.01, kind)))
            g, ref = c.greeks(S, 0.03, sigma), func_lib.bs_greeks(S, 100.0, 0.25, 0.03, sigma, 0.0, kind)
            self.assertEqual(list(g), list(ref))
            for k in g:
                self.assertTrue(np.array_equal(g[k], ref[k]))
            price = float(c.price(101.0, 0.03, 0.22))
            self.assertEqual(c.implied_vol(price, 101.0, 0.03),
                             func_lib.implied_vol(price, 101.0, 100.0, 0.25, 0.03, 0.0, kind))
            self.assertAlmostEqual(c.implied_vol(price, 101.0, 0.03), 0.22, places=6)
        self.assertEqual(OptionContract.from_days('put', 100, 21).T, 21 / func_lib.TRADING_DAYS)
        self.assertTrue(np.array_equal(OptionContract('call', 100, 1).payoff([90, 110]), [0, 10]))
        self.assertTrue(np.array_equal(OptionContract('put', 100, 1).payoff([90, 110]), [10, 0]))
        with self.assertRaises(ValueError):
            OptionContract('calls', 100, 1)


class TestTidyUps(unittest.TestCase):
    def test_snake_case_names(self):
        self.assertIs(func_lib.createHistPrices, func_lib.create_hist_prices)
        self.assertIs(func_lib.computingReturns, func_lib.compute_returns)
        self.assertIs(quantlib.cross_section.computingReturns, quantlib.cross_section.compute_returns)
        prices = pd.DataFrame(100 + np.arange(40.0).reshape(20, 2), index=pd.bdate_range('2020-01-01', periods=20),
                              columns=['A', 'B'])
        pd.testing.assert_frame_equal(func_lib.computingReturns(prices, [1, 5]), func_lib.compute_returns(prices, [1, 5]))

    def test_train_agent_importable_from_both(self):
        self.assertIs(quantlib.clustering.train_agent, quantlib.rl.train_agent)
        self.assertIs(func_lib.train_agent, quantlib.rl.train_agent)

    def test_rsi_functions_unchanged_in_kind(self):
        close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(size=60)))
        self.assertEqual(len(func_lib.wilder_rsi(close)), len(close))
        self.assertLess(len(func_lib.calculate_rsi(close.diff())), len(close))


if __name__ == '__main__':
    unittest.main()
