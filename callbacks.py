from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error

def register_callbacks(app, df, selected_features_ridge, model_ridge_best, x_test, y_test, test_mape):
    @app.callback(
        Output('model-metrics', 'children'),
        Input('url', 'pathname')
    )
    def update_metrics(_):
        y_pred = model_ridge_best.predict(x_test[selected_features_ridge])
        mae = mean_absolute_error(y_test, y_pred)
        
        return html.Div([
            html.H4("Model Description"),
            html.P("This dashboard uses a Ridge Regression model trained on 1-year lagged economic indicators to predict government debt levels."),
            html.H4("Model Performance"),
            html.P(f"Mean Absolute Percentage Error: {test_mape:.2f}%"),
        ])

    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [State('country-selector', 'value')] +
        [State(f'input-{col}', 'value') for col in selected_features_ridge]
    )
    def predict_debt(n_clicks, country, *input_values):
        if n_clicks is None:
            return ""
        
        if any(v is None for v in input_values):
            return html.Div("Please provide values for all model inputs before predicting.", 
                          style={'color': 'red'})

        input_data = pd.DataFrame([input_values], columns=selected_features_ridge)
        prediction = model_ridge_best.predict(input_data)[0]

        return html.Div([
            html.H4(f"Predicted Debt as Percent of GDP for {country}:"),
            html.P(f"{prediction:.2f}% of GDP", className='prediction-value'),
            html.P("Note: Adjust economic indicators above to change the prediction.", className='prediction-note')
        ])

    @app.callback(
        [Output(f'input-{col}', 'value') for col in selected_features_ridge],
        Input('country-selector', 'value')
    )
    def update_inputs(country):
        latest_row = df[df['OECD'] == country].sort_values('Year').iloc[-1]
        return [latest_row.get(col, None) for col in selected_features_ridge]

    @app.callback(
        Output('time-series-plot', 'figure'),
        Input('country-selector', 'value')
    )
    def update_time_series(country):
        filtered = df[df['OECD'] == country]
        fig = px.line(
            filtered,
            x='Year',
            y=["Central Government Debt (Percent of GDP)", "Predicted_Debt"],
            title=f"Historical Government Debt: {country}",
            labels={
                'Year': 'Year',
                'value': 'Debt (% of GDP)',
                'variable': 'Type'
            },
            color_discrete_sequence=['#0074D9', '#FF7F0E']
        )
        
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
            legend_title_text='Data Type',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.for_each_trace(lambda t: t.update(name='Actual' if t.name == "Central Government Debt (Percent of GDP)" else 'Predicted'))
        
        return fig