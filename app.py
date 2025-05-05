import dash
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset_to_model_with.csv"))

# Data processing and model setup
lagged_df = df[[col for col in df.columns if col in ['OECD', 'Year', 'Central Government Debt (Percent of GDP)'] or col.endswith('_lag1')]]
lagged_df = lagged_df[lagged_df.Year != 2010]

train_df = lagged_df[lagged_df['Year'] < 2023]
test_df = lagged_df[lagged_df['Year'] == 2023]

x_train = train_df[[col for col in train_df.columns if col.endswith('_lag1')]]
y_train = train_df["Central Government Debt (Percent of GDP)"]

x_test = test_df[[col for col in test_df.columns if col.endswith('_lag1')]]
y_test = test_df["Central Government Debt (Percent of GDP)"]

selected_features_ridge = [
    'Central Government Debt (Percent of GDP)_lag1', 
    'LongInterestRate_lag1',
    'GovernmentExpenditure_Housing and community amenities_lag1',
    'GovernmentExpenditure_Fuel and energy_lag1',
    'GovernmentExpenditure_Education_lag1',
    'GovernmentExpenditure_General economic, commercial and labour affairs_lag1',
    'GovernmentExpenditure_Health_lag1'
]

model_ridge_best = Ridge(alpha=11.233240329780312).fit(x_train[selected_features_ridge], y_train)
y_pred_ridge = model_ridge_best.predict(x_test[selected_features_ridge])

y_test_safe = np.where(y_test == 0, np.finfo(float).eps, y_test)
test_mape = np.mean(np.abs((y_test_safe - y_pred_ridge) / y_test_safe)) * 100

# Generate predictions for all years
lagged_df['Predicted_Debt'] = model_ridge_best.predict(lagged_df[selected_features_ridge])
lagged_df['Difference'] = lagged_df['Predicted_Debt'] - lagged_df['Central Government Debt (Percent of GDP)']

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets')
server = app.server

# Set layout with the dataframe containing predictions
app.layout = create_layout(lagged_df, selected_features_ridge)

# Register callbacks with the enhanced dataframe
register_callbacks(app, lagged_df, selected_features_ridge, model_ridge_best, x_test, y_test, test_mape)

if __name__ == '__main__':
    app.run(debug=True)