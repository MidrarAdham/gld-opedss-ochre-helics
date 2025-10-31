import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, callback, Output, Input
from datetime import datetime

app = Dash(__name__)


app.layout = html.Div ([
    html.H1(children='Real-Time House Data from OCHRE',style= {'textAlign':'center'}),
    dcc.Graph(id="Live-graph"),
    dcc.Graph(id="Live-graph-2"),
    dcc.Interval (id="tick", interval=500, n_intervals=0) # refereshes every half a second
])

@callback (
        Output("Live-graph", "figure"),
        Output("Live-graph-2", "figure"),
        Input("tick", "n_intervals")
        )

def update_graph (_):
    try:

        df = pd.read_csv('./test.csv', header=None, names=["label", "value"])
    except Exception:
        return go.Figure(layout=dict(title="Reading ..."))
    
    if df.empty:
        return go.Figure(layout=dict(title="No data just yet ..."))
    
    if len(df) > 200:
        df = df.tail(200).reset_index(drop=True)
        
    # x = list(range(len(df)))
    # x = df["label"].to_list()
    o = 500
    x = pd.date_range(end=pd.Timestamp(datetime.now()), periods=len(df), freq=f'{o}ms')
    y = df["value"]
    # good = ~y.isna()
    # x,y = x[good], y[good]

    x_end = x.max()
    x_start = x_end - pd.Timedelta(seconds=60)

    mask = (x >= x_start) & (x <= x_end)

    x, y = x[mask], y[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="value"))
    fig.update_layout (
        title="Live Data from the OCHRE Load",
        xaxis_title="Timestamp (hh:mm:ss)",
        yaxis_title="Apparent Power (VA)",

        margin=dict(l=30, r=10, t=40, b=30)
    )
    fig.update_xaxes(range=[x_start, x_end])

    # Ading another plot:

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="lines", name="value"))
    fig2.update_layout (
        title="Live Data from GridLAB-D Load",
        xaxis_title="Timestamp (hh:mm:ss)",
        yaxis_title="Apparent Power (VA)",
        margin=dict(l=30, r=10, t=40, b=30)
    )
    fig2.update_xaxes(range=[x_start, x_end])
    return fig, fig2

if __name__ == "__main__":
    app.run(debug=True)
