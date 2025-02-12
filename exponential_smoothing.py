from statsmodels.tsa.holtwinters import ExponentialSmoothing , SimpleExpSmoothing

def plot_forecast(forecast , name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'] , y = train['#Passengers'] , name = "train"))
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast , name = name))   
    fig.show()      


model = SimpleExpSmoothing(train['#Passengers'] )
model_fit = model.fit(smoothing_level=0.2, optimized=True)
forecast = model_fit.forecast(len(test))

plot_forecast(forecast , 'exponential smoothing')




model1 = ExponentialSmoothing(train['#Passengers'], trend='add', seasonal='add', seasonal_periods=12)
model1_fit = model1.fit(smoothing_level=0.2, smoothing_trend=0.2, smoothing_seasonal=0.2 , optimized=True)
forecast1 = model1_fit.forecast(len(test))

plot_forecast(forecast1 , 'exponential smoothing with trend and seasonal')






