
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
import numpy as np
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


def chart_behave(final_dict):
    try:
        df = final_dict["resultados"]["dataframe"]
        x = ['status_quo', 'aversion_perdida', 'sensibilidad_decreciente']
        sq_si = float(df.status_quo.values[0].strip("%"))/100*df.ocurrencias.values[0]
        sq_no = df.ocurrencias.values[0] - sq_si
        av_si = float(df.aversion_perdida.values[0].strip("%"))/100*df.ocurrencias.values[0]
        av_no = df.ocurrencias.values[0] - av_si

        keys_oc = [key for key in final_dict["ocurrencias"].keys()]
        del keys_oc[0]
        res = []
        for occ in keys_oc:
            cond1 = final_dict["ocurrencias"][occ]["operaciones"]["ganadoras"]["profit_ganadora"] > 0
            cond2 = final_dict["ocurrencias"][occ]["operaciones"]["ganadoras"]["profit_ganadora"] > final_dict["ocurrencias"][keys_oc[0]]["operaciones"]["ganadoras"]["profit_ganadora"] and final_dict["ocurrencias"][occ]["operaciones"]["perdedoras"]["profit_perdedora"] > final_dict["ocurrencias"][keys_oc[0]]["operaciones"]["perdedoras"]["profit_perdedora"]
            cond3 = final_dict["ocurrencias"][occ]["operaciones"]["perdedoras"]["profit_perdedora"] / final_dict["ocurrencias"][occ]["operaciones"]["ganadoras"]["profit_ganadora"] > 2
            res.append((cond1 and cond2) or (cond3 and cond2) or (cond3 and cond1))
        res = [1 if x == "True" else 0 for x in res]

        dec_si = np.array(res).sum()
        dec_no = df.ocurrencias.values[0] - dec_si

        y1 = [sq_si, av_si, dec_si]
        y2 = [sq_no, av_no, dec_no]

        data = {'labels': x,
                'Si': y1,
                'No': y2
                }

        df_p = pd.DataFrame(data)

        xcoord = [0,1,2]

        annotations1 = [dict(
                    x=xi-0.2,
                    y=yi,
                    text=str(yi),
                    xanchor='auto',
                    yanchor='bottom',
                    showarrow=False,
                ) for xi, yi in zip(xcoord, y1)]

        annotations2 = [dict(
                    x=xi+0.2,
                    y=yi,
                    text=str(yi),
                    xanchor='auto',
                    yanchor='bottom',
                    showarrow=False,
                ) for xi, yi in zip(xcoord, y2)]

        annotations = annotations1 + annotations2

        trace1 = go.Bar(
            x=x,
            y=y1,
            name='Sí'
        )
        trace2 = go.Bar(
            x=x,
            y=y2,
            name='No'
        )
        data = [trace1, trace2]
        layout = go.Layout(
            barmode='group',
            annotations=annotations
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
    except:
        print("sin ocurrencias")
