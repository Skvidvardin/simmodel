import flask
import base64
import io

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import plotly.graph_objects as go

# import qdc
from simmodel_app import qdc  # for DEBUG

FONT_AWESOME = "./assets/fontawesome-free-5.13.0-web/css/all.css"

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG, FONT_AWESOME])
app.config.suppress_callback_exceptions = True

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25%",
    "padding": "2rem 1rem",
    "background-color": "#111111",
    "overflow": "auto"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25%",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebarFastTab = dcc.Tab(label='Fast settings', children=html.Div([

    # SIDEBAR START/END DATE OF SIMULATION
    html.Div(
        [html.Br(),
         dbc.Row([
             dbc.Col(
                 [html.P("Start date:", style={'color': '#cccccc'}),
                  dcc.Input(
                      id='start-day-input',
                      placeholder='DD/MM/YYYY',
                      type='text',
                      value=(datetime.datetime.today() - relativedelta(years=1)).strftime(
                          '%d/%m/%Y'),
                      style={'backgroundColor': "#222222", 'borderColor': "#222222",
                             'color': '#cccccc'}
                  )], width='auto'
             ),
             dbc.Col(
                 [html.P("End Date:", style={'color': '#cccccc'}),
                  dcc.Input(
                      id='end-day-input',
                      placeholder='DD/MM/YYYY',
                      type='text',
                      value=datetime.datetime.today().strftime('%d/%m/%Y'),
                      style={'backgroundColor': "#222222", 'borderColor': "#222222",
                             'color': '#cccccc'}
                  )], width='auto'
             ),
         ]
         )]
    ),
    html.Hr(),

    # SIDEBAR N SIMUlATIONS
    html.Div(
        [html.P("Select number of simulations:", style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-simulations-input',
                     placeholder=10,
                     type='tel',
                     value=10,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-simulations-slider',
                     min=0,
                     max=1000,
                     step=10,
                     value=10,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         )]
    ),
    html.Hr(),

    # SIDEBAR N WORKERS
    html.Div(
        [html.P("Select number of workers:", style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-workers-input',
                     placeholder=50,
                     type='tel',
                     value=50,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-workers-slider',
                     min=0,
                     max=1000,
                     step=10,
                     value=50,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         ),
         html.P(),
         html.P(html.Span(["And upload deltas if required ",
                           html.Span(html.I(className="fas fa-exclamation-triangle red"),
                                     style={'color': "#5bc0de"}
                                     ),
                           ":"]),
                style={'color': '#cccccc'},
                id='warningIcon'),
         dbc.Tooltip(
             dcc.Markdown('''
                                            Please, upload a CSV file with data about changes/deltas in number of 
                                            workers;
                                            
                                            CSV must contain two comma-separated columns 'date' and 'delta'. Values of
                                             'date' must be business days in dd/mm/yyyy format. Values of 'delta' must
                                              be positive or negative integers;
                                              
                                            Warning: minimal interval between changes in number of workers should be
                                             less than any model execution. All models will be adjusted.
                                              Use "Advanced settings" tab for more precise model.
                                            '''),
             target=f"warningIcon",
             placement='right',
             style={'font-size': '200%', 'textAlign': 'left', }),
         dcc.Upload(
             id='upload-data',
             children=html.Div([
                 'Drag and Drop or Click and Select Files',
             ]),
             multiple=False,
             style={'width': '100%',
                    'height': '10%',
                    'lineHeight': '200%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '2px',
                    'borderColor': '#444444',
                    'textAlign': 'center',
                    }),
         html.Div(id='output-data-upload', style={'color': '#444444'}),
         html.Div(id='output-data-upload-hidden', style={'display': 'none'}),

         ]
    ),
    html.Hr(),

    # SIDEBAR MODELS INCOME
    html.Div(
        [html.P("Select number of income models by month:", style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-models-income-input',
                     placeholder=100,
                     type='tel',
                     value=100,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-models-income-slider',
                     min=0,
                     max=1000,
                     step=10,
                     value=100,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         )]
    ),

    html.P(),
    html.P(html.Span(["Or upload data by day ",
                      html.Span(html.I(className="fas fa-exclamation-triangle"),
                                style={'color': "#5bc0de"}
                                ),
                      ":"]),
           style={'color': '#cccccc'},
           id='warningIcon2'),
    dbc.Tooltip(
        dcc.Markdown('''
                                            Please, upload a CSV file with data about number of models;

                                            CSV must contain two comma-separated columns 'date' and 'volume'. Values of
                                             'date' must be in dd/mm/yyyy format. Values of 'volume' must be positive
                                              integers;
                                              
                                            Please, verify that models come only at business days.
                                            '''),
        target=f"warningIcon2",
        placement='right',
        style={'font-size': '200%', 'textAlign': 'left', }),
    dcc.Upload(
        id='upload-data2',
        children=html.Div([
            'Drag and Drop or Click and Select Files',
        ]),
        multiple=False,
        style={'width': '100%',
               'height': '10%',
               'lineHeight': '200%',
               'borderWidth': '2px',
               'borderStyle': 'dashed',
               'borderRadius': '2px',
               'borderColor': '#444444',
               'textAlign': 'center',
               }),
    html.Div(id='output-data-upload2', style={'color': '#444444'}),
    html.Div(id='output-data-upload-hidden2', style={'display': 'none'}),
    html.Hr(),

    # SIDEBAR MODELS IN QUEUE
    html.Div(
        [html.P("Select number of models in queue already:", style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-models-in-queue-input',
                     placeholder=100,
                     type='tel',
                     value=100,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-models-in-queue-slider',
                     min=0,
                     max=1000,
                     step=10,
                     value=100,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         )]
    ),
    html.P(),
    # SIDEBAR MODELS IN PROGRESS
    html.Div(
        [html.P("Select number of models in progress already:",
                style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-models-in-progress-input',
                     placeholder=100,
                     type='tel',
                     value=50,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-models-in-progress-slider',
                     min=0,
                     max=1000,
                     step=10,
                     value=50,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         )]
    ),
    html.Hr(),

    # SIDEBAR AVG/STD DAYS PER MODEL
    html.Div(
        [dbc.Row([
            dbc.Col(
                [html.P("Average days per one model:", style={'color': '#cccccc'}),
                 dcc.Input(
                     id='avg-days-per-model-input',
                     placeholder=14,
                     type='tel',
                     value=14,
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 )], width='auto'
            ),
            dbc.Col(
                [html.P("Std days per one model:", style={'color': '#cccccc'}),
                 dcc.Input(
                     id='std-days-per-model-input',
                     placeholder=2,
                     type='tel',
                     value=2,
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 )], width='auto'
            ),
        ]
        )]
    ),
    html.Hr(),
    html.Div(
        [html.P(style={'color': '#FF0000'}, id='warning')],

    ),

    # SUBMIT BOTTOM
    # html.Br(),

    dbc.Button(
        "Submit",
        color="primary",
        block=True,
        id="button",
        className="mb-3",
        n_clicks=0
    )]))

sidebarAdvancedTab = dcc.Tab(label='Advanced settings', children=html.Div([

    # SIDEBAR START/END DATE OF SIMULATION
    html.Div(
        [html.Br(),
         dbc.Row([
             dbc.Col(
                 [html.P("Start date:", style={'color': '#cccccc'}),
                  dcc.Input(
                      id='start-day-input-advanced',
                      placeholder='DD/MM/YYYY',
                      type='text',
                      value=(datetime.datetime.today() - relativedelta(years=1)).strftime(
                          '%d/%m/%Y'),
                      style={'backgroundColor': "#222222", 'borderColor': "#222222",
                             'color': '#cccccc'}
                  )], width='auto'
             ),
             dbc.Col(
                 [html.P("End Date:", style={'color': '#cccccc'}),
                  dcc.Input(
                      id='end-day-input-advanced',
                      placeholder='DD/MM/YYYY',
                      type='text',
                      value=datetime.datetime.today().strftime('%d/%m/%Y'),
                      style={'backgroundColor': "#222222", 'borderColor': "#222222",
                             'color': '#cccccc'}
                  )], width='auto'
             ),
         ]
         )]
    ),
    html.Hr(),

    # SIDEBAR N SIMUlATIONS
    html.Div(
        [html.P("Select number of simulations:", style={'color': '#cccccc'}),
         dbc.Row([
             dbc.Col(
                 dcc.Input(
                     id='n-simulations-input-advanced',
                     placeholder=10,
                     type='tel',
                     value=10,
                     size='2',
                     style={'backgroundColor': "#222222", 'borderColor': "#222222",
                            'color': '#cccccc'}
                 ), width='auto'
             ),
             dbc.Col(
                 dcc.Slider(
                     id='n-simulations-slider-advanced',
                     min=0,
                     max=1000,
                     step=10,
                     value=10,
                     marks={
                         0: '0',
                         200: '200',
                         400: '400',
                         600: '600',
                         800: '800',
                         1000: '1000',
                     },
                 )
             )
         ]
         )]
    ),
    html.Hr(),

    # SIDEBAR MODELS MATRIX
    html.Div([
        html.P('Average days per one model:', style={'color': '#cccccc'}),
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='sidebar-table-1',
                    columns=[{'name': 'worker/model type', 'id': 'column-type', 'editable': False, 'renamable': False},
                             {'name': 'BaseModel', 'id': 'BaseModel', 'deletable': True, 'renamable': False},
                             ],
                    data=[{'column-type': 'BaseWorker', 'BaseModel': '14'}],
                    editable=True,
                    row_deletable=True,
                    style_header={'backgroundColor': '#222222',
                                  'border': '1px solid #cccccc'},
                    style_cell={
                        'backgroundColor': '#111111',
                        'color': '#cccccc',
                        'border': '1px solid #cccccc'},
                    style_data_conditional=[{
                        'if': {'column_editable': False},
                        'backgroundColor': '#222222',
                        'color': '#cccccc'
                    }],
                    style_header_conditional=[{
                        'if': {'column_editable': False},
                        'backgroundColor': '#222222',
                        'color': '#cccccc'
                    }],
                    css=[{'selector': 'td.cell--selected, td.focused', 'rule': 'background-color: #111111 !important;'},
                         {'selector': 'td.cell--selected *, td.focused *', 'rule': 'color: #cccccc !important;'}]
                ),
            ], width=11),
        ], justify="center"),

        html.P(),
        html.P('Std days per one model:', style={'color': '#cccccc'}),
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='sidebar-table-2',
                    columns=[{'name': 'worker/model type', 'id': 'column-type', 'editable': False, 'renamable': False},
                             {'name': 'BaseModel', 'id': 'BaseModel', 'deletable': True, 'renamable': False},
                             ],
                    data=[{'column-type': 'BaseWorker', 'BaseModel': '2'}],
                    editable=True,
                    row_deletable=True,
                    style_header={'backgroundColor': '#222222',
                                  'border': '1px solid #cccccc'},
                    style_cell={
                        'backgroundColor': '#111111',
                        'color': '#cccccc',
                        'border': '1px solid #cccccc'},
                    style_data_conditional=[{
                        'if': {'column_editable': False},
                        'backgroundColor': '#222222',
                        'color': '#cccccc'
                    }],
                    style_header_conditional=[{
                        'if': {'column_editable': False},
                        'backgroundColor': '#222222',
                        'color': '#cccccc'
                    }],
                    css=[{'selector': 'td.cell--selected, td.focused', 'rule': 'background-color: #111111 !important;'},
                         {'selector': 'td.cell--selected *, td.focused *', 'rule': 'color: #cccccc !important;'}]
                ),
            ], width=11),
        ], justify="center"),

        html.P(),
        dbc.Row([
            dbc.Col([
                dcc.Input(
                    id='adding-row-input',
                    placeholder='Enter worker type...',
                    value='',
                    size='14',
                    style={'backgroundColor': "#222222", 'borderColor': "#222222",
                           'color': '#cccccc'}
                ),
                dbc.Button(html.I(className="fas fa-plus-circle fa-2x", style={'color': "#5bc0de"}),
                           n_clicks=0, id='adding-row-button', color="link"),
            ]),
            dbc.Col([
                dcc.Input(
                    id='adding-column-input',
                    placeholder='Enter model type...',
                    value='',
                    size='14',
                    style={'backgroundColor': "#222222", 'borderColor': "#222222",
                           'color': '#cccccc'}
                ),
                dbc.Button(html.I(className="fas fa-plus-circle fa-2x", style={'color': "#5bc0de"}),
                           n_clicks=0, id='adding-column-button', color="link"),
            ])
        ]),
    ]),

    html.Hr(),

    # SIDEBAR N WORKERS
    html.P(html.Span(["Upload number of workers ",
                      html.Span(html.I(className="fas fa-exclamation-triangle"),
                                style={'color': "#5bc0de"}
                                ),
                      ":"]),
           style={'color': '#cccccc'},
           id='warningIcon-advanced'),
    dbc.Tooltip(
        dcc.Markdown('''
             Please, upload a CSV file with data about number of workers by types;

             CSV must contain comma-separated columns 'date', 'worker_type_1', 'worker_type_2', etc. 
             Where 'worker_type_1', 'worker_type_2', etc. are your types of workers.
             Values of 'date' must be business days in dd/mm/yyyy format.
                                            '''),
        target=f"warningIcon-advanced",
        placement='right',
        style={'font-size': '200%', 'textAlign': 'left', }),
    dcc.Upload(
        id='upload-data-advanced',
        children=html.Div([
            'Drag and Drop or Click and Select Files',
        ]),
        multiple=False,
        style={'width': '100%',
               'height': '10%',
               'lineHeight': '200%',
               'borderWidth': '2px',
               'borderStyle': 'dashed',
               'borderRadius': '2px',
               'borderColor': '#444444',
               'textAlign': 'center',
               }),
    html.Div(id='output-data-upload-advanced', style={'color': '#444444'}),
    html.Div(id='output-data-upload-hidden-advanced', style={'display': 'none'}),
    html.Hr(),

    # SIDEBAR MODELS INCOME
    html.P(html.Span(["Upload income models ",
                      html.Span(html.I(className="fas fa-exclamation-triangle"),
                                style={'color': "#5bc0de"}
                                ),
                      ":"]),
           style={'color': '#cccccc'},
           id='warningIcon2-advanced'),
    dbc.Tooltip(
        dcc.Markdown('''
        Please, upload a CSV file with data about number of models;

        CSV must contain comma-separated columns 'date', 'model_type_1', 'model_type_2', etc. 
        Where 'worker_type_1', 'worker_type_2', etc. are your types of workers.
        Values of 'date' must be in dd/mm/yyyy format.

        Please, verify that models come only at business days.
        '''),
        target=f"warningIcon2-advanced",
        placement='right',
        style={'font-size': '200%', 'textAlign': 'left', }),
    dcc.Upload(
        id='upload-data2-advanced',
        children=html.Div([
            'Drag and Drop or Click and Select Files',
        ]),
        multiple=False,
        style={'width': '100%',
               'height': '10%',
               'lineHeight': '200%',
               'borderWidth': '2px',
               'borderStyle': 'dashed',
               'borderRadius': '2px',
               'borderColor': '#444444',
               'textAlign': 'center',
               }),
    html.Div(id='output-data-upload2-advanced', style={'color': '#444444'}),
    html.Div(id='output-data-upload-hidden2-advanced', style={'display': 'none'}),
    html.Hr(),

    # SIDEBAR MODELS IN QUEUE
    html.P("Select number of models in queue/progress already:", style={'color': '#cccccc'}),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='sidebar-table-3',
                columns=[{'name': 'model type', 'id': 'column-type', 'editable': False, 'renamable': False},
                         {'name': 'in progress', 'id': 'in-progress', 'deletable': False, 'renamable': False},
                         {'name': 'in queue', 'id': 'in-queue', 'deletable': False, 'renamable': False},
                         ],
                data=[{'column-type': 'BaseModel', 'in-progress': '50', 'in-queue': '150'}],
                editable=True,
                row_deletable=True,
                style_header={'backgroundColor': '#222222',
                              'border': '1px solid #cccccc'},
                style_cell={
                    'backgroundColor': '#111111',
                    'color': '#cccccc',
                    'border': '1px solid #cccccc'},
                style_data_conditional=[{
                    'if': {'column_editable': False},
                    'backgroundColor': '#222222',
                    'color': '#cccccc'
                }],
                style_header_conditional=[{
                    'if': {'column_editable': False},
                    'backgroundColor': '#222222',
                    'color': '#cccccc'
                }],
                css=[{'selector': 'td.cell--selected, td.focused', 'rule': 'background-color: #111111 !important;'},
                     {'selector': 'td.cell--selected *, td.focused *', 'rule': 'color: #cccccc !important;'}]
            ),
        ], width=11),
    ], justify="center"),

    html.P(),
    dbc.Row([
        dbc.Col([
            dcc.Input(
                id='adding-row-input-2',
                placeholder='Enter model type...',
                value='',
                size='14',
                style={'backgroundColor': "#222222", 'borderColor': "#222222",
                       'color': '#cccccc'}
            ),
            dbc.Button(html.I(className="fas fa-plus-circle fa-2x", style={'color': "#5bc0de"}),
                       n_clicks=0, id='adding-row-button-2', color="link"),
        ]),
        dbc.Col('', width=6)
    ]),

    html.Hr(),

    # SIDEBAR WARNINGS
    html.Div(html.P(style={'color': '#FF0000'}, id='warning-advanced')),

    # SUBMIT BUTTON
    # html.Br(),

    dbc.Button(
        "Submit",
        color="primary",
        block=True,
        id="button-advanced",
        className="mb-3",
        n_clicks=0
    )]))

contentArea = html.Div(
    children=dcc.Loading([html.Div(id='main-content')], style={'position': 'fixed', 'left': '62.5%', 'top': '40%'}),
    style=CONTENT_STYLE, id='content'
)

app.layout = html.Div([dcc.Location(id="url"),

                       # SIDEBAR HEADER
                       html.Div([html.H2("Simmodel Dashboard", className="display-4", style={'color': '#cccccc'}),
                                 html.Div([dcc.Tabs([sidebarFastTab, sidebarAdvancedTab])])],
                                style=SIDEBAR_STYLE),

                       html.Div(id='n-clicks-catcher', style={'display': 'none'}),
                       html.Div(id='n-clicks-catcher-advanced', style={'display': 'none'}),

                       # CONTENT
                       contentArea

                       ],
                      )


@app.callback(
    dash.dependencies.Output('n-simulations-input', 'value'),
    [dash.dependencies.Input('n-simulations-slider', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('n-simulations-input-advanced', 'value'),
    [dash.dependencies.Input('n-simulations-slider-advanced', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('n-workers-input', 'value'),
    [dash.dependencies.Input('n-workers-slider', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('n-models-income-input', 'value'),
    [dash.dependencies.Input('n-models-income-slider', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('n-models-in-queue-input', 'value'),
    [dash.dependencies.Input('n-models-in-queue-slider', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('n-models-in-progress-input', 'value'),
    [dash.dependencies.Input('n-models-in-progress-slider', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('warning', 'children'),
    [dash.dependencies.Input('n-models-in-progress-input', 'value'),
     dash.dependencies.Input('n-workers-input', 'value')])
def update_output(value1, value2):
    value = ''
    if int(value1) > int(value2):
        value = f"Please, validate that initial in-progress size ({value1}) <= number of workers ({value2})"
    return value


@app.callback([Output("main-content", "children"),
               Output("n-clicks-catcher", "children"),
               Output("n-clicks-catcher-advanced", "children")],

              [Input('button', 'n_clicks'),
               Input('button-advanced', 'n_clicks')],

              [State('n-clicks-catcher', 'children'),
               State('n-clicks-catcher-advanced', 'children'),

               State('start-day-input', 'value'),
               State('end-day-input', 'value'),
               State('n-simulations-input', 'value'),
               State('n-workers-input', 'value'),
               State('output-data-upload-hidden', 'children'),
               State('n-models-income-input', 'value'),
               State('output-data-upload-hidden2', 'children'),
               State('n-models-in-queue-input', 'value'),
               State('n-models-in-progress-input', 'value'),
               State('avg-days-per-model-input', 'value'),
               State('std-days-per-model-input', 'value'),

               State('start-day-input-advanced', 'value'),
               State('end-day-input-advanced', 'value'),
               State('n-simulations-input-advanced', 'value'),
               State('sidebar-table-1', 'data'),
               State('sidebar-table-2', 'data'),
               State('output-data-upload-hidden-advanced', 'children'),
               State('output-data-upload-hidden2-advanced', 'children'),
               State('sidebar-table-3', 'data'),
               ])
def update_output(n_clicks, n_clicks_advanced,

                  n_clicks_catched, n_clicks_catched_advanced,

                  startDate, endDate, simulationsNum, nWorkers, nWorkersDict,
                  modelsIncome, modelsIncomeDict, initialQueueSize, initialInprogressSize,
                  avgWorkDaysPerModel, sdWorkDaysPerModel,

                  startDateAdvanced, endDateAdvanced, simulationsNumAdvanced,
                  avgModelMartixAdvanced, stdModelMartixAdvanced,
                  workersNumDictAdvanced, modelsIncomeDictAdvanced,
                  initialQueueInprogressMatrixAdvanced
                  ):

    if n_clicks_catched is None and n_clicks_catched_advanced is None:
        n_clicks_catched, n_clicks_catched_advanced = 0, 0

    if n_clicks > n_clicks_catched:
        basic_model_flag = True
        advanced_model_flag = False
    elif n_clicks_advanced > n_clicks_catched_advanced:
        basic_model_flag = False
        advanced_model_flag = True
    else:
        basic_model_flag = False
        advanced_model_flag = False

    n_clicks_catched = n_clicks
    n_clicks_catched_advanced = n_clicks_advanced

    # BASIC MODEL
    if advanced_model_flag is False:
        if basic_model_flag is True and advanced_model_flag is False:

            try:
                if eval(nWorkersDict) is not None:
                    nWorkersDF = pd.DataFrame(eval(nWorkersDict))
                else:
                    nWorkersDF = None
            except:
                nWorkersDF = None

            try:
                if eval(modelsIncomeDict) is not None:
                    modelsIncomeDF = pd.DataFrame(eval(modelsIncomeDict))
                else:
                    modelsIncomeDF = None
            except:
                nWorkersDF = None

            df = qdc.sm_main(int(simulationsNum), int(nWorkers), nWorkersDF,
                             int(modelsIncome), modelsIncomeDF, int(initialQueueSize), int(initialInprogressSize),
                             int(avgWorkDaysPerModel), float(sdWorkDaysPerModel),
                             datetime.datetime.strptime(startDate, '%d/%m/%Y'),
                             datetime.datetime.strptime(endDate, '%d/%m/%Y'),
                             )

        elif basic_model_flag is False and advanced_model_flag is False:
            df = pd.read_csv('./assets/sample_df.csv', sep=',')

        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#cccccc'},
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#222222', zerolinecolor='#222222')

        )

        # GRAPH 1 (Models Dynamic)
        fig1 = go.Figure(data=[
            go.Bar(name='Income models',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'incomeNum'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#0275d8', line=dict(width=0)),
                   ),
            go.Bar(name='Models in Queue',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'queueNum'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#d9534f', line=dict(width=0)),
                   ),
            go.Bar(name='Models in progress',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'inProgressNum'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#5bc0de', line=dict(width=0)),
                   ),
            go.Bar(name='Done models',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'doneNum'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#5cb85c', line=dict(width=0)),
                   ),
        ], layout=layout)
        fig1.update_layout(title_text='Models Dynamic (average)', title_x=0.5)

        # GRAPH 2 (Queue Dynamics)
        fig2 = go.Figure(data=[go.Box(y=df.loc[df['date'] == date, 'queueNum'],
                                      name=datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y'),
                                      line=dict(color='#0275d8'))
                               for date in df.date.unique()],
                         layout=layout)
        fig2.update_layout(title_text='Queue Dynamics', title_x=0.5)

        # GRAPH 3 (Average Models Dynamics)
        fig3 = go.Figure(data=[
            go.Bar(name='Waiting Time',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'avgWaitingTime'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#0275d8', line=dict(width=0)),
                   ),
            go.Bar(name='Serving Time',
                   x=[datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y') for date in df.date.unique()],
                   y=[df.loc[df['date'] == date, 'avgServingTime'].mean() for date in df.date.unique()],
                   marker=go.bar.Marker(color='#d9534f', line=dict(width=0)),
                   ),
        ], layout=layout)
        fig3.update_layout(barmode='stack')
        fig3.update_layout(title_text='Models Dynamics (average)', title_x=0.5)

        # GRAPH 4 (Models Time Till Done)
        fig4 = go.Figure(data=[go.Box(y=df.loc[df['date'] == date, 'avgTime2Done'],
                                      name=datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y'),
                                      line=dict(color='#0275d8'))
                               for date in df.date.unique()],
                         layout=layout)
        fig4.update_layout(title_text='Average Time2Done', title_x=0.5)

        main_content = [dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig1)
            ], width='6'),
            dbc.Col([
                dcc.Graph(figure=fig2)
            ], width='6'),
        ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig3)
                ], width='6'),
                dbc.Col([
                    dcc.Graph(figure=fig4)
                ], width='6'),
            ])]

    # ADVANCED MODEL
    elif advanced_model_flag is True and basic_model_flag is False:

        main_content = [dbc.Row([
            dbc.Col([
                dcc.Graph()
            ], width='6'),
            dbc.Col([
                dcc.Graph()
            ], width='6'),
        ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph()
                ], width='6'),
                dbc.Col([
                    dcc.Graph()
                ], width='6'),
            ])]

        avgModelMartixAdvanced = pd.DataFrame(avgModelMartixAdvanced).rename(columns={'column-type': 'WorkerType'})
        stdModelMartixAdvanced = pd.DataFrame(stdModelMartixAdvanced).rename(columns={'column-type': 'WorkerType'})
        initialQueueInprogressMatrixAdvanced = pd.DataFrame(initialQueueInprogressMatrixAdvanced)\
            .rename(columns={'column-type': 'ModelType', 'in-progress': 'InProgress', 'in-queue': 'InQueue'})

        workersNumDFAdvanced = workersNumDictAdvanced
        modelsIncomeDFAdvanced = modelsIncomeDictAdvanced

        df = qdc.sm_advanced_main(datetime.datetime.strptime(startDateAdvanced, '%d/%m/%Y'),
                                  datetime.datetime.strptime(endDateAdvanced, '%d/%m/%Y'),
                                  int(simulationsNumAdvanced),
                                  avgModelMartixAdvanced, stdModelMartixAdvanced,
                                  workersNumDFAdvanced, modelsIncomeDFAdvanced,
                                  initialQueueInprogressMatrixAdvanced)

    else:
        main_content = ''

    return main_content, n_clicks_catched, n_clicks_catched_advanced


@app.callback([Output('output-data-upload', 'children'),
               Output('output-data-upload-hidden', 'children')
               ],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents, filename, last_modified):
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return 'Successfully uploaded: ' + filename, str(df.to_dict())

        except Exception as e:
            return 'There was an error processing this file, try more.', 'None'
    else:
        return '', 'None'


@app.callback([Output('output-data-upload2', 'children'),
               Output('output-data-upload-hidden2', 'children')
               ],
              [Input('upload-data2', 'contents')],
              [State('upload-data2', 'filename'),
               State('upload-data2', 'last_modified')])
def update_output(contents, filename, last_modified):
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return 'Successfully uploaded: ' + filename, str(df.to_dict())

        except Exception as e:
            return 'There was an error processing this file, try more.', 'None'
    else:
        return '', 'None'


@app.callback(
    [Output('sidebar-table-1', 'data'),
     Output('sidebar-table-2', 'data')],
    [Input('adding-row-button', 'n_clicks')],
    [State('adding-row-input', 'value'),
     State('sidebar-table-1', 'data'),
     State('sidebar-table-1', 'columns'),
     State('sidebar-table-2', 'data'),
     State('sidebar-table-2', 'columns')
     ])
def add_row(n_clicks, row_input, existing_rows_1, existing_columns_1, existing_rows_2, existing_columns_2):
    if n_clicks > 0:
        d1, d2 = {}, {}
        for c in existing_columns_1:
            if c['id'] == 'column-type':
                d1[c['id']] = row_input
            else:
                d1[c['id']] = ''
        for c in existing_columns_2:
            if c['id'] == 'column-type':
                d2[c['id']] = row_input
            else:
                d2[c['id']] = ''
        existing_rows_1.append(d1)
        existing_rows_2.append(d2)
    return existing_rows_1, existing_rows_2


@app.callback(
    [Output('sidebar-table-1', 'columns'),
     Output('sidebar-table-2', 'columns')],
    [Input('adding-column-button', 'n_clicks')],
    [State('adding-column-input', 'value'),
     State('sidebar-table-1', 'columns'),
     State('sidebar-table-2', 'columns')])
def update_columns(n_clicks, value, existing_columns_1, existing_columns_2):
    if n_clicks > 0:
        existing_columns_1.append({
            'id': value, 'name': value,
            'renamable': False, 'deletable': True
        })
        existing_columns_2 = existing_columns_1
    return existing_columns_1, existing_columns_2


@app.callback(
    Output('sidebar-table-3', 'data'),
    [Input('adding-row-button-2', 'n_clicks')],
    [State('adding-row-input-2', 'value'),
     State('sidebar-table-3', 'data'),
     State('sidebar-table-3', 'columns')])
def update_columns(n_clicks, row_input, existing_rows_1, existing_columns_1):
    if n_clicks > 0:
        d1 = {}
        for c in existing_columns_1:
            if c['id'] == 'column-type':
                d1[c['id']] = row_input
            else:
                d1[c['id']] = ''
        existing_rows_1.append(d1)
    return existing_rows_1


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)
