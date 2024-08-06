"""
Digital Filter Design App
Mauricio Martinez-Garcia
Copyright 2024
"""

import dash
from dash import Dash, dcc, callback, Input, Output, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import numpy as np
import pandas as pd
import io

app = dash.Dash(__name__)

filter_types = pd.DataFrame({'family': ['Butterworth', 'Chebyshev - I', 'Chebyshev - II']})

app.layout = html.Div(
    [
        html.H1("Digital Filter Design"),
        html.Hr(),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(filter_types.family, 'Butterworth', id='dropdown-selection'),
                ], style={'margin-bottom': '12px'}),
                html.Div([
                    'Order',
                    dcc.Slider(id="filter-order", min=0, max=6, value=2, step=1),
                ], style={'margin-bottom': '12px'}),
                html.Div([
                    'Cutoff frequency: ',
                    dcc.Input(id="fc-input", type="number", min=0, max=1, value=0.5),
                    dcc.Slider(id='fc-slider', min=0, max=1, value=0.5),
                    ], style={'margin-bottom': '12px'}),
                html.Div(
                    [
                        'Ripple (dB)',
                        dcc.Slider(id='ripple-slider', min=1, max=5, value=1),
                        ], style={'display': 'none'}, id='ripple-div'),
                html.Div(
                    [
                        'Attenuation (dB)',
                        dcc.Slider(id='atten-slider', min=10, max=90, value=10),
                        ], style={'width': '95%', 'display': 'none'}, id='atten-div'),

            ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}),
            html.Div([
                dcc.Graph(id="frequency-plot"),
            ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'middle'}),
        ]),
        dcc.Store(id='filter-data')
    ],
)


@callback(
    Output('filter-data', 'data'),
    Output('fc-slider', 'value'),
    Output('fc-input', 'value'),
    Output('ripple-div', 'style'),
    Output('atten-div', 'style'),
    Input('dropdown-selection', 'value'),
    Input('filter-order', 'value'),
    Input('fc-slider', 'value'),
    Input('fc-input', 'value'),
    Input('ripple-slider', 'value'),
    Input('atten-slider', 'value'),
)
def get_filter(filter_family, filter_order, fc_slider, fc_input, ripple, atten):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == 'fc-slider':
        fc = fc_slider
    else:
        fc = fc_input
    if filter_family == 'Butterworth':
        b, a = signal.butter(filter_order, fc)
        has_ripple = 'none'
        has_atten = 'none'
    elif filter_family == 'Chebyshev - I':
        b, a = signal.cheby1(filter_order, ripple, fc)
        has_ripple = 'block'
        has_atten = 'none'
    else:
        b, a = signal.cheby2(filter_order, atten, fc)
        has_ripple = 'none'
        has_atten = 'block'
    w, h = signal.freqz(b, a)
    df = pd.DataFrame({'Frequency': w,
                       'Gain': 20 * np.log10(abs(h)),
                       'Phase': np.arctan2(np.imag(h), np.real(h))}
                      )
    return df.to_json(), fc, fc, dict(display=has_ripple), dict(display=has_atten)


@callback(
    Output('frequency-plot', 'figure'),
    Input('filter-data', 'data')
)
def update_graph(jsonified_data):
    df = pd.read_json(io.StringIO(jsonified_data))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=df['Frequency'], y=df['Gain'], name="Gain (dB)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['Frequency'], y=df['Phase'], name="Phase (rad)"),
        row=2, col=1,
    )
    fig['layout']['xaxis2']['title'] = 'Frequency &#969; (rad/sample)'
    fig.update_yaxes(range=[-75, 5], row=1, col=1)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
