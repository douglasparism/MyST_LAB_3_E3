
"""
# -- --------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                           -- #
# -- script: visualizations.py : python script with data visualization fun -- #
# -- author: YOUR GITHUB USER NAME                                         -- #
# -- license: GPL-3.0 License                                              -- #
# -- repository: YOUR REPOSITORY URL                                       -- #
# -- --------------------------------------------------------------------- -- #
"""
import plotly.express as px
import pandas as pd
from typing import Dict
import plotly.graph_objects as go


def pie_chart(df_ranking: pd.DataFrame, *args, **kwargs) -> px.pie:
    fig = px.pie(df_ranking, values='rank', names='Symbol', *args, **kwargs)
    return fig


def line_chart(
    profits: pd.DataFrame,
    df: pd.DataFrame,
    *args,
    **kwargs
) -> px.line:

    draws = pd.DataFrame(
        df.iloc[[2, 3, 5, 6], 2].map(
            lambda x: profits[profits['timestamp'] == x]
            ['profit_acm_d'].values[0]
        ).values,
        df.iloc[[2, 3, 5, 6], 2].values
    )

    draws.reset_index(inplace=True)
    draws.set_axis(['date', 'value'], axis=1, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=profits.loc[:, 'timestamp'],
                   y=profits.loc[:, 'profit_acm_d'],
                   mode='lines', name='Profit')
    )
    fig.add_trace(
        go.Scatter(x=draws.loc[:, 'date'].values[:2],
                   y=draws.loc[:, 'value'].values[:2],
                   mode='lines+markers', name='Drawdown')
    )
    fig.add_trace(
        go.Scatter(x=draws.loc[:, 'date'].values[2:],
                   y=draws.loc[:, 'value'].values[2:],
                   mode='lines+markers', name='Drawup')
    )
    return fig


def bar_chart() -> px.bar:
    fig = px.bar()
    return fig
