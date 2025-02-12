import pandas as pd
import plotly.graph_objects as go

# Ensure 'Month' is in datetime format
train['Month'] = pd.to_datetime(train['Month'])
test['Month'] = pd.to_datetime(test['Month'])

# Extract month number
train['month_number'] = train['Month'].dt.month
test['month_number'] = test['Month'].dt.month

# Seasonal Naive Forecasting
snaive_fc = test['month_number'].map(lambda m: train[train['month_number'] == m]['#Passengers'].iloc[-1])

# Function to plot forecast
def plot_forecast(forecast, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name="Train"))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecast, name=name, mode='lines+markers'))
    
    fig.update_layout(title=name, xaxis_title="Month", yaxis_title="Passengers")
    fig.show()

# Plot the forecast
plot_forecast(snaive_fc, 'Seasonal Naive Forecast')
