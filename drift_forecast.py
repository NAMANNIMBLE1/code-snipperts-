constant = (train['#Passengers'].iloc[-1] - train['#Passengers'].iloc[0]) / (len(train) - 1)

# Use len(test) instead of len(train) to generate correct h values
test['h'] = range(len(test))

# Compute the drift forecast correctly
test['drift_forecast'] = train['#Passengers'].iloc[-1] + constant * test['h']

def plot_forecast(forecast , name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'], y = train['#Passengers'], name="train"))
    fig.add_trace(go.Scatter(x = test['Month'], y = forecast, name=name))
    fig.show()  

plot_forecast(test['drift_forecast'], 'drift forecast')
