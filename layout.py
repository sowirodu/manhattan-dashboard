from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Dictionary of variable descriptions for tooltips
VARIABLE_DESCRIPTIONS = {
    'Central Government Debt (Percent of GDP)_lag1': 'Previous year\'s central government debt as percentage of GDP',
    'LongInterestRate_lag1': 'Previous year\'s long-term interest rate (%)',
    'GovernmentExpenditure_Housing and community amenities_lag1': 'Previous year\'s govt spending on housing/community',
    'GovernmentExpenditure_Fuel and energy_lag1': 'Previous year\'s govt spending on fuel and energy',
    'GovernmentExpenditure_Education_lag1': 'Previous year\'s govt spending on education',
    'GovernmentExpenditure_General economic, commercial and labour affairs_lag1': 'Previous year\'s govt spending on economic affairs',
    'GovernmentExpenditure_Health_lag1': 'Previous year\'s govt spending on health'
}

def create_layout(df, selected_features_ridge):
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.H1("OECD Countries Debt as Percent of GDP Dashboard", className='header'),
        
        html.Div([
            html.H3("Ridge Regression Model"),
            html.Div(id='model-metrics')
        ], className='metrics-container'),
        
        html.Div([
            html.H3("Make New Predictions"),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='country-selector',
                        options=[{'label': c, 'value': c} for c in df['OECD'].unique()],
                        value='United States',
                        style={'margin-bottom': '20px'}
                    ),
                    html.Button('Predict Debt', id='predict-button', className='predict-button')
                ], className='country-selector-container'),
                
                html.Div([
                    html.Div([
                        html.Div([
                            create_input_field(col) for col in selected_features_ridge
                        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'padding': '10px'})
                    ])
                ], className='indicators-container')
            ], className='prediction-interface')
        ], className='prediction-container'),
        
        html.Div(id='prediction-output', className='prediction-output'),
        
        html.Div([
            dcc.Tabs([
                dcc.Tab(label='Time Series', children=[
                    dcc.Graph(id='time-series-plot')
                ]),
                dcc.Tab(label='Raw Data', children=[
                    dash_table.DataTable(
                        id='data-table',
                        columns=[
                            {"name": "Year", "id": "Year"},
                            {"name": "Actual Debt (% GDP)", "id": "Central Government Debt (Percent of GDP)"},
                            {"name": "Predicted Debt (% GDP)", "id": "Predicted_Debt"},
                            {"name": "Difference", "id": "Difference"}
                        ],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{Difference} > 0',
                                    'column_id': 'Difference'
                                },
                                'backgroundColor': '#FFCCCC',
                                'color': 'black'
                            },
                            {
                                'if': {
                                    'filter_query': '{Difference} < 0',
                                    'column_id': 'Difference'
                                },
                                'backgroundColor': '#CCFFCC',
                                'color': 'black'
                            }
                        ]
                    )
                ])
            ])
        ], className='data-explorer-container')
    ])

def create_input_field(col):
    is_key_predictor = col in ['Central Government Debt (Percent of GDP)_lag1', 'LongInterestRate_lag1']
    
    return html.Div([
        html.Div([
            html.Label(col, style={'margin-bottom': '5px', 'font-weight': 'bold'}),
            html.Span(" 🔑", style={'color': '#FFD700', 'margin-left': '5px'}) if is_key_predictor else None,
            html.Span(" ⓘ", className='tooltip-icon', id=f'tooltip-target-{col}'),
            dbc.Tooltip(
                VARIABLE_DESCRIPTIONS.get(col, 'No description available'),
                target=f'tooltip-target-{col}',
                placement='right'
            )
        ], style={'display': 'flex', 'align-items': 'center'}),
        dcc.Input(
            id=f'input-{col}',
            type='number',
            step='any',
            style={
                'width': '100%', 
                'padding': '8px',
                'border': '2px solid #FFD700' if is_key_predictor else '1px solid #ddd',
                'border-radius': '4px'
            }
        )
    ], style={
        'margin': '15px',
        'flex': '1 0 250px',
        'min-width': '250px',
        'background-color': '#FFF9E6' if is_key_predictor else '#f9f9f9',
        'padding': '15px',
        'border-radius': '8px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
    })