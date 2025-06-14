from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error

def register_callbacks(app, lagged_df, df, selected_features_ridge, model_ridge_best, x_test, y_test, test_mape):
    @app.callback(
        Output('model-metrics', 'children'),
        Input('url', 'pathname')
    )
    def update_metrics(_):
        y_pred = model_ridge_best.predict(x_test[selected_features_ridge])
        mae = mean_absolute_error(y_test, y_pred)
        
        return html.Div([
            html.H4("Model Description"),
            html.P("This dashboard uses a Ridge Regression model trained on 1-year lagged economic indicators to predict Debt-to-GDP ratio."),
            html.H4("Model Performance"),
            html.P(f"Mean Absolute Percentage Error: {test_mape:.2f}%"),
        ])

    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [State('country-selector', 'value')] +
        [State(f'input-{col}', 'value') for col in selected_features_ridge]
    )
    # In callbacks.py, modify the predict_debt function
    def predict_debt(n_clicks, country, *input_values):
        if n_clicks is None:
            return ""
        
        if any(v is None for v in input_values):
            return html.Div("Please provide values for all model inputs before predicting.", 
                        style={'color': 'red'})

        input_data = pd.DataFrame([input_values], columns=selected_features_ridge)
        prediction = model_ridge_best.predict(input_data)[0]
        
        # Update the 2024 prediction in the dataframe
        country_mask = (lagged_df['OECD'] == country) & (lagged_df['Year'] == 2024)
        if country_mask.any():
            lagged_df.loc[country_mask, 'Predicted_Debt'] = prediction
        
        return html.Div([
            html.H4(f"Predicted Debt-to-GDP Ratio for {country} in 2024:"),
            html.P(f"{prediction:.2f}% of GDP", className='prediction-value'),
            html.P("Note: This prediction will be reflected in the time series chart.", className='prediction-note')
        ])

    @app.callback(
        [Output(f'input-{col}', 'value') for col in selected_features_ridge],
        Input('country-selector', 'value')
    )
    def update_inputs(country):
        # Get the latest row from the original dataframe
        latest_row = df[df['OECD'] == country].sort_values('Year').iloc[-1]
        
        # Create mapping from lagged feature names to their non-lagged counterparts
        feature_mapping = {
            'Central Government Debt (Percent of GDP)_lag1': 'Central Government Debt (Percent of GDP)',
            'LongInterestRate_lag1': 'LongInterestRate',
            'GovernmentExpenditure_Housing and community amenities_lag1': 'GovernmentExpenditure_Housing and community amenities',
            'GovernmentExpenditure_Fuel and energy_lag1': 'GovernmentExpenditure_Fuel and energy',
            'GovernmentExpenditure_Education_lag1': 'GovernmentExpenditure_Education',
            'GovernmentExpenditure_General economic, commercial and labour affairs_lag1': 'GovernmentExpenditure_General economic, commercial and labour affairs',
            'GovernmentExpenditure_Health_lag1': 'GovernmentExpenditure_Health'
        }
        
        # Get values - use non-lagged values where available, otherwise use the lagged ones
        values = []
        for col in selected_features_ridge:
            non_lagged_col = feature_mapping.get(col, col)
            if non_lagged_col in df.columns:
                values.append(latest_row.get(non_lagged_col, None))
            else:
                values.append(latest_row.get(col, None))
        
        return values

    @app.callback(
        Output('time-series-plot', 'figure'),
        Input('country-selector', 'value')
    )
    # modify the update_time_series function
    def update_time_series(country):
        filtered = lagged_df[lagged_df['OECD'] == country]
        
        # Create figure with existing data
        fig = px.line(
            filtered,
            x='Year',
            y=["Central Government Debt (Percent of GDP)", "Predicted_Debt"],
            title=f"Historical and Projected Government Debt-to-GDP Ratio: {country}",
            labels={
                'Year': 'Year',
                'value': 'Debt-to-GDP Ratio',
                'variable': 'Type'
            },
            color_discrete_sequence=['#0074D9', '#FF7F0E']
        )
        
        # Add marker for 2024 prediction if it exists
        if 2024 in filtered['Year'].values:
            pred_2024 = filtered[filtered['Year'] == 2024]['Predicted_Debt'].values[0]
            fig.add_scatter(
                x=[2024],
                y=[pred_2024],
                mode='markers',
                marker=dict(color='#FF7F0E', size=12),
                name='2024 Projection',
                showlegend=True
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
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)',
                gridwidth=0.5,
                tickmode='linear',
                tick0=filtered['Year'].min(),
                dtick=1,
                range=[filtered['Year'].min(), 2025]  # Adjust as needed
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)',
                gridwidth=0.5
            )
        )
        
        fig.for_each_trace(lambda t: t.update(
            name='Actual' if t.name == "Central Government Debt (Percent of GDP)" else 'Predicted' if t.name != "2024 Projection" else t.name
        ))
        
        return fig