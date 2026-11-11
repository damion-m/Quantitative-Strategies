"""Section 4 of the original func_lib: clustering helpers (and the generic train_agent loop that sat there).
"""

import numpy as np
from plotly.subplots import make_subplots

from .theme import DOWN, NEUTRAL, PALETTE


# =============================================================================
# 4. Clustering helpers
# =============================================================================
def find_elbow_point(sse, k_range):
    """Elbow = the k whose (normalized) SSE point lies farthest from the first-to-last chord."""
    sse = np.asarray(sse, dtype=float)
    k = np.asarray(k_range, dtype=float)
    sse_normalized = (sse - sse.min()) / (sse.max() - sse.min())
    k_normalized = (k - k.min()) / (k.max() - k.min())

    # Perpendicular distance to the chord, vectorized (2-D cross product written out:
    # np.cross on 2-D vectors is deprecated in NumPy 2)
    p1 = np.array([k_normalized[0], sse_normalized[0]])
    d = np.array([k_normalized[-1], sse_normalized[-1]]) - p1
    distances = np.abs(d[0] * (sse_normalized - p1[1]) - d[1] * (k_normalized - p1[0])) / np.linalg.norm(d)

    elbow_idx = int(np.argmax(distances))
    optimal_k = k_range[elbow_idx]
    print(f'Optimal number of clusters: {optimal_k}')
    return optimal_k, list(distances), k_normalized, sse_normalized


def plot_optimal_cluster_point(sse, k_range):
    optimal_k, distances, k_normalized, sse_normalized = find_elbow_point(sse, k_range)
    elbow_idx = list(k_range).index(optimal_k)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Normalized SSE and Distance to Chord'))
    fig.add_scatter(x=list(k_range), y=list(sse), mode='lines+markers', name='SSE',
                    line=dict(color=PALETTE[0]), row=1, col=1)
    fig.add_scatter(x=k_normalized, y=sse_normalized, mode='lines+markers', name='Normalized SSE',
                    line=dict(color=PALETTE[0]), row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[sse_normalized[0], sse_normalized[-1]], mode='lines', name='Reference line',
                    line=dict(color=PALETTE[1], dash='dash'), row=1, col=2)

    # Perpendicular connectors from each point to the chord
    p1 = np.array([0, sse_normalized[0]])
    v = np.array([1, sse_normalized[-1]]) - p1
    for xk, yk in zip(k_normalized, sse_normalized):
        proj = p1 + np.dot(np.array([xk, yk]) - p1, v) / np.dot(v, v) * v
        fig.add_scatter(x=[xk, proj[0]], y=[yk, proj[1]], mode='lines', showlegend=False, hoverinfo='skip',
                        line=dict(color=NEUTRAL, dash='dot', width=1), row=1, col=2)

    # Highlight the elbow (the original highlighted argmax(SSE), i.e. always the first k)
    for col, x, y in [(1, k_range[elbow_idx], sse[elbow_idx]), (2, k_normalized[elbow_idx], sse_normalized[elbow_idx])]:
        fig.add_scatter(x=[x], y=[y], mode='markers', name=f'Elbow (k = {optimal_k})', showlegend=col == 1,
                        marker=dict(color=DOWN, size=14, line=dict(color='white', width=1.5)), row=1, col=col)
    fig.update_xaxes(title_text='Number of Clusters', row=1, col=1)
    fig.update_yaxes(title_text='Sum of Squared Errors', row=1, col=1)
    fig.update_xaxes(title_text='Normalized Number of Clusters', row=1, col=2)
    fig.update_yaxes(title_text='Normalized SSE', row=1, col=2)
    fig.update_layout(height=460, hovermode='closest')
    fig.show()


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
