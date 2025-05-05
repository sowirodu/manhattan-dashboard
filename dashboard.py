import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import plotly.express as px

# Importing nesseceray packages
import statsmodels.api as sm
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error


# Load data and clean column names
df = pd.read_csv('dataset_to_model_with_3.csv')


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

app.layout = html.Div([
    html.H1("OECD Economic Data Explorer", style={'textAlign': 'center'}),
    
    # 1. Country and Year Selectors
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='country-selector',
                options=[{'label': country, 'value': country} for country in df['OECD'].unique()],
                value='United States',
                clearable=False
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.RangeSlider(
                id='year-slider',
                min=df['Year'].min(),
                max=df['Year'].max(),
                value=[df['Year'].min(), df['Year'].max()],
                marks={str(year): str(year) for year in range(df['Year'].min(), df['Year'].max()+1, 5)},
                step=1
            )
        ], style={'width': '48%', 'float': 'right', 'padding': '20px 0'})
    ]),
    
    # 2. Main Visualization Tabs
    dcc.Tabs([
        # Tab 1: Time Series Explorer
        dcc.Tab(label='Time Series', children=[
            html.Div([
                dcc.Dropdown(
                    id='y-axis-selector',
                    options=[{'label': col, 'value': col} 
                           for col in df.columns if col not in ['OECD', 'Year', 'DebtQuantile']],
                    value='Central Government Debt (Percent of GDP)',
                    multi=False
                ),
                dcc.Graph(id='time-series-plot')
            ])
        ]),
        
        # Tab 2: Correlation Analysis
        dcc.Tab(label='Correlations', children=[
            html.Div([
                dcc.Dropdown(
                    id='corr-var1',
                    options=[{'label': col, 'value': col} 
                           for col in df.columns if df[col].dtype in ['float64', 'int64']],
                    value='GDP (current US$)'
                ),
                dcc.Dropdown(
                    id='corr-var2',
                    options=[{'label': col, 'value': col} 
                           for col in df.columns if df[col].dtype in ['float64', 'int64']],
                    value='Tax revenue (% of GDP)'
                ),
                dcc.Graph(id='scatter-plot')
            ])
        ]),
        
        # Tab 3: Raw Data
        dcc.Tab(label='Raw Data', children=[
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                page_size=15,
                style_table={'overflowX': 'auto'},
                filter_action="native",
                sort_action="native"
            )
        ])
    ])
])

# Callback for time series plot
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('country-selector', 'value'),
     Input('year-slider', 'value'),
     Input('y-axis-selector', 'value')]
)
def update_time_series(country, years, y_axis):
    filtered = df[(df['OECD'] == country) & 
                 (df['Year'] >= years[0]) & 
                 (df['Year'] <= years[1])]
    
    fig = px.line(filtered, x='Year', y=y_axis,
                  title=f"{country}: {y_axis} ({years[0]}-{years[1]})")
    fig.update_layout(hovermode="x unified")
    return fig

# Callback for scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('country-selector', 'value'),
     Input('corr-var1', 'value'),
     Input('corr-var2', 'value')]
)
def update_scatter(country, var1, var2):
    filtered = df[df['OECD'] == country]
    fig = px.scatter(filtered, x=var1, y=var2, color='Year',
                     hover_data=['Year'],
                     title=f"{country}: {var1} vs {var2}")
    fig.update_traces(marker=dict(size=10))
    return fig

if __name__ == '__main__':
    app.run(debug=True)