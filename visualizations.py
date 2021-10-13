
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


def pie_chart(df_ranking: pd.DataFrame) -> px.pie:
    fig = px.pie(df_ranking, values=df_ranking['rank'],
                 names=df_ranking['Symbol'])
    return fig


def line_chart() -> px.line:
    fig = px.line()
    return fig


def bar_chart() -> px.bar:
    fig = px.bar()
    return fig
