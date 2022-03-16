from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
from pandas_datareader import data
import numpy as np
import seaborn as sns

app = Dash(__name__)

colors = {
    'background': '#103d5c',
    'text': '#FFFFFF'
}

theme = {
    "accent":"#60bab6",
    "accent_positive":"#357e94",
    "accent_negative":"#C20000",
    "background_content":"#242f42",
    "background_page":"#20293d",
    "body_text":"white",
    "border":"#242f42",
    "border_style":{
        "name":"underlined",
        "borderWidth":"0px 0px 1px 0px",
        "borderStyle":"solid",
        "borderRadius":0
    },
    "button_border":{
        "width":"1px",
        "color":"#60bab6",
        "radius":"5px"
    },
    "button_capitalization":"capitalize",
    "button_text":"#242f42",
    "button_background_color":"#60bab6",
    "control_border":{
        "width":"0px",
        "color":"#242f42",
        "radius":"0px"
    },
    "control_background_color":"#20293d",
    "control_text":"white",
    "card_margin":"15px",
    "card_padding":"5px",
    "card_border":{
        "width":"0px 0px 0px 0px",
        "style":"solid",
        "color":"#242f42",
        "radius":"0px"
    },
    "card_background_color":"#242f42",
    "card_box_shadow":"0px 0px 0px rgba(0,0,0,0)",
    "card_outline":{
        "width":"0px",
        "style":"solid",
        "color":"#242f42"
    },
    "card_header_margin":"0px",
    "card_header_padding":"10px",
    "card_header_border":{
        "width":"0px 0px 1px 0px",
        "style":"solid",
        "color":"#242f42",
        "radius":"0px"
    },
    "card_header_background_color":"#242f42",
    "card_header_box_shadow":"0px 0px 0px rgba(0,0,0,0)",
    "breakpoint_font":"1200px",
    "breakpoint_stack_blocks":"700px",
    "colorway":[
        "#60bab6",
        "#3f4f75",
        "#80cfbe",
        "#f4564e",
        "#ffeeb2",
        "#20293d",
        "#faddd2",
        "#ffdd68",
        "#357e94",
        "#a1acc3"
    ],
    "colorscale":[
        "#ffffff",
        "#f0f0f0",
        "#d9d9d9",
        "#bdbdbd",
        "#969696",
        "#737373",
        "#525252",
        "#252525",
        "#000000"
    ],
    "dbc_primary":"#60bab6",
    "dbc_secondary":"#6b83ae",
    "dbc_info":"#3BA8C3",
    "dbc_gray":"#adb5bd",
    "dbc_success":"#00CCA4",
    "dbc_warning":"#FADD6A",
    "dbc_danger":"#F76065",
    "font_family":"Open Sans",
    "font_family_header":"Open Sans",
    "font_family_headings":"Open Sans",
    "font_size":"17px",
    "font_size_smaller_screen":"15px",
    "font_size_header":"24px",
    "title_capitalization":"uppercase",
    "header_content_alignment":"spread",
    "header_margin":"0px 0px 15px 0px",
    "header_padding":"0px",
    "header_border":{
        "width":"0px 0px 0px 0px",
        "style":"solid",
        "color":"#242f42",
        "radius":"0px"
    },
    "header_background_color":"#242f42",
    "header_box_shadow":"none",
    "header_text":"white",
    "heading_text":"white",
    "text":"white",
    "report_background_content":"#FAFBFC",
    "report_background_page":"white",
    "report_text":"black",
    "report_font_family":"Computer Modern",
    "report_font_size":"12px"
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'font-family':'Montserrat','backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

     html.Div(children=[
        html.Label('Dropdown'),
        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),

        html.Br(),
        html.Label('Multi-Select Dropdown'),
        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
                     ['Montréal', 'San Francisco'],
                     multi=True),

        html.Br(),
        html.Label('Radio Items'),
        dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
    ], style={'color': colors['text'],'padding': 10, 'flex': 1}),

    

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)