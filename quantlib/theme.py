"""Section 1 of the original func_lib: Plotly dark theme and small chart helpers.

Importing this module registers the quant_dark template and makes it the default.
"""

import plotly.graph_objects as go
import plotly.io as pio


# =============================================================================
# 1. Plotly dark theme
# =============================================================================
BG = '#0e1117'
PALETTE = ['#4CC9F0', '#F72585', '#43AA8B', '#F9C74F', '#9D4EDD', '#F8961E', '#90BE6D']
UP, DOWN, NEUTRAL = '#43AA8B', '#F94144', '#8d99ae'

RANGE_SELECTOR = dict(buttons=[
    dict(count=1, label='1M', step='month', stepmode='backward'),
    dict(count=6, label='6M', step='month', stepmode='backward'),
    dict(count=1, label='YTD', step='year', stepmode='todate'),
    dict(count=1, label='1Y', step='year', stepmode='backward'),
    dict(step='all', label='All'),
], bgcolor='#1b1f2a', activecolor='#3a3f4b', font=dict(color='#e6e6e6'),
    x=0, xanchor='left', y=1.02, yanchor='bottom')   # top-left, just above the plot; the legend is at the bottom

THEME_MARGIN = dict(l=60, r=30, t=110, b=50)   # t: room for title + subtitle above the range buttons


def set_plotly_theme():
    """Register the 'quant_dark' template and make it the default for all figures."""
    pio.templates['quant_dark'] = go.layout.Template(layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family='Inter, "Segoe UI", Helvetica, Arial, sans-serif', size=13, color='#e6e6e6'),
        # Title pinned to the top edge, so a subtitle line grows downward into the margin rather than
        # into the range buttons / legend / subplot titles that sit just above the plot area
        title=dict(font=dict(size=20, color='#ffffff'), x=0.01, xanchor='left',
                   yref='container', y=1, yanchor='top', pad=dict(t=14),
                   subtitle=dict(font=dict(size=13, color='#9aa3b2'))),
        colorway=PALETTE,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1b1f2a', bordercolor='#3a3f4b', font=dict(color='#e6e6e6')),
        # Legend centred along the bottom edge of the figure. At the top it collides with the range buttons,
        # subplot titles and long titles once the window is narrow; down here Plotly grows the bottom
        # margin to fit it, so it can wrap onto as many rows as it needs at any width.
        legend=dict(orientation='h', yref='container', y=0, yanchor='bottom', xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)'),
        margin=THEME_MARGIN,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', zeroline=False,
                   showline=True, linecolor='#3a3f4b'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', zeroline=False,
                   showline=True, linecolor='#3a3f4b'),
    ))
    pio.templates.default = 'plotly_dark+quant_dark'


set_plotly_theme()   # applied on import so every func_lib chart is themed


def subtitled(title, subtitle):
    """A chart title with a smaller subtitle line underneath, e.g. title=subtitled('GBM edge', 'value / price − 1').

    Plotly can't wrap a long title, so in a narrow window it gets cut off. Keep the headline short
    and put the detail (parameters, dates, how to read the chart) in the subtitle.
    """
    return dict(text=title, subtitle=dict(text=subtitle))


def has_subtitle(title):
    return isinstance(title, dict) and bool(title.get('subtitle', {}).get('text'))


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'
