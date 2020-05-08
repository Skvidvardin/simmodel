import flask

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import plotly.graph_objects as go

import simmodel
# from simmodel_app import simmodel  # for DEBUG

# def generate_table(dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])
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

app.layout = html.Div([dcc.Location(id="url"),
                       # SIDEBAR
                       html.Div(
                           [
                               # SIDEBAR HEADER
                               html.H2("Simmodel Dashboard", className="display-4", style={'color': '#cccccc'}),
                               html.Hr(),

                               # SIDEBAR START/END DATE OF SIMULATION
                               html.Div(
                                   [dbc.Row([
                                       dbc.Col(
                                           [html.P("Start date:", style={'color': '#cccccc'}),
                                            dcc.Input(
                                                id='start-day-input',
                                                placeholder='DD/MM/YYYY',
                                                type='text',
                                                value=(datetime.datetime.today() - relativedelta(years=1)).strftime('%d/%m/%Y'),
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
                                    )]
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
                                   [html.P("Select number of models in progress already:", style={'color': '#cccccc'}),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Input(
                                                id='n-models-in-progress-input',
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
                                                id='n-models-in-progress-slider',
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

                               # SUBMIT BOTTOM
                               html.Br(),
                               dbc.Button(
                                   "Submit",
                                   color="primary",
                                   block=True,
                                   id="button",
                                   className="mb-3",
                               ),
                           ],
                           style=SIDEBAR_STYLE,
                       ),

                       # CONTENT
                       html.Div(children=dcc.Loading([
                           dbc.Row([
                               dbc.Col([
                                   dcc.Graph(id='graph-1')
                               ], width='6'),
                               dbc.Col([
                                   dcc.Graph(id='graph-2')
                               ], width='6'),
                           ]),
                           dbc.Row([
                               dbc.Col([
                                   dcc.Graph(id='graph-3')
                               ], width='6'),
                               dbc.Col([
                                   dcc.Graph(id='graph-4')
                               ], width='6'),
                           ]),
                       ], style={'position': 'fixed', 'left': '62.5%', 'top': '40%'}),
                           style=CONTENT_STYLE, id='content'
                       ),

                       ],
                      )


@app.callback(
    dash.dependencies.Output('n-simulations-input', 'value'),
    [dash.dependencies.Input('n-simulations-slider', 'value')])
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


@app.callback([Output("graph-1", "figure"),
               Output("graph-2", "figure"),
               Output("graph-3", "figure"),
               Output("graph-4", "figure"),
               ],
              [Input('button', 'n_clicks')],
              [State('start-day-input', 'value'),
               State('end-day-input', 'value'),
               State('n-simulations-input', 'value'),
               State('n-workers-input', 'value'),
               State('n-models-income-input', 'value'),
               State('n-models-in-queue-input', 'value'),
               State('n-models-in-progress-input', 'value'),
               State('avg-days-per-model-input', 'value'),
               State('std-days-per-model-input', 'value'),])
def update_output(n_clicks, startDate, endDate, simulationsNum, nWorkers,
                  modelsIncome, initialQueueSize, initialInprogressSize,
                  avgWorkDaysPerModel, sdWorkDaysPerModel):

    if n_clicks is not None:
        df = simmodel.sm_main(int(simulationsNum), int(nWorkers),
                              int(modelsIncome), int(initialQueueSize), int(initialInprogressSize),
                              int(avgWorkDaysPerModel), float(sdWorkDaysPerModel),
                              datetime.datetime.strptime(startDate, '%d/%m/%Y'),
                              datetime.datetime.strptime(endDate, '%d/%m/%Y'),
                              )
    else:
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

    #GRAPH 3 (Average Models Dynamics)
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

    #GRAPH 4 (Models Time Till Done)
    fig4 = go.Figure(data=[go.Box(y=df.loc[df['date'] == date, 'avgTime2Done'],
                                  name=datetime.datetime.strptime(date, '%m-%Y').strftime('%b-%Y'),
                                  line=dict(color='#0275d8'))
                           for date in df.date.unique()],
                     layout=layout)

    return fig1, fig2, fig3, fig4


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=8000)
