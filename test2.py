# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_cool_components
import dash
import dash_bootstrap_components as dbc
import datetime
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    
app.layout = html.Div(
    [
        html.H1('Stock Funds Analysis', style = {'textAlign': 'center'}),
#         dbc.Row([dbc.Col(html.H3("Stock Analysis"),
#                        width= {'size':6, 'offset':3},
#                        )]
#                ),
        
        dbc.Row([
                dbc.Col(
                        dcc.Input(id="input", value="", placeholder="add value..."),
                        
                        ),
                dbc.Col(
                        html.Button("Add Option", id="submit", n_clicks=0),
                        ),
                dbc.Col(
                        dcc.Dropdown([
                        {}], multi = True, id="dropdown"),
                         width = {'size':3, 'offset' : 2},
                        ),
                       

                dbc.Col(
                        dbc.Button("Show Funds", id="my-button", n_clicks=0),
                        style = {"paddingLeft" : 0},
                )
                ]),
                
            dcc.Graph(id='my_graph',
            figure={'data':[
            {'x':[1,2], 'y':[3,1]}

            ], }), 
        
            html.Div(id="slider", children = []),
            html.Div(id="slider_output", children = []),
                
            
                
    ]
)

#this callback is for: Adding user input into the dropdown list
@app.callback(
    Output("dropdown", "options"),
    Input("submit", "n_clicks"),
    State("input", "value"),
    State("dropdown", "options"),
    prevent_initial_call=True,
)
def add_dropdown_option(n_clicks, input_value, options):
    return options + [{"label": input_value, "value": input_value}]

#this callback is for: displaying the graph based on user input
@app.callback(
    Output(component_id='my_graph', component_property='figure'),
    
    [Input('my-button','n_clicks')],
    [State('dropdown','value')],prevent_initial_call=True
)
def display_test(n,input_data):
    fig = go.Figure()
    start = '2017-01-01'
    end = '2021-12-31'
    df = data.DataReader(input_data, 'yahoo', start, end)
    adj_close = df["Adj Close"]
    ret_series = ((1 + adj_close.pct_change()).cumprod() -1)
    fig = px.line(ret_series, y=ret_series.columns, title='Life expectancy in Canada')
    return fig

#this callback is for: Based on user input, create a slider
@app.callback(
    Output('slider','children'),
    [Input('dropdown','value')],prevent_initial_call=True
)
def slider(input_data1):
    children = []
    for i in range (len(input_data1)):
                    new_label = html.Label(input_data1[i])
                    new_slider = dcc.Slider(
                    0.1,1,0.1, id = input_data1[i])
                    children.append(new_label)
                    children.append(new_slider)
    return children


    
    
if __name__ == '__main__':
    app.run_server(debug=False)
