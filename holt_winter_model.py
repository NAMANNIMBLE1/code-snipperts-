from statsmodels.tsa.holtwinters import SimpleExpSmoothing , Holt , ExponentialSmoothing

def plot_forecast(forecast , forecast1 , forecast2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'] , y = train['#Passengers'] , name = "train"))
    fig.add_trace(go.Scatter(x = test['Month'] , y = test['#Passengers'] , name = 'test'))
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast , name = "simple"))     
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast1 , name = 'holt linear')) 
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast2 , name = 'holt winter'))       
    return fig.show()   


model1 = Holt(train['#Passengers'])
model_fit = model.fit()
forecast1 = model_fit.forecast(len(test))

model2 = ExponentialSmoothing(train['#Passengers'], trend='add', seasonal='add', seasonal_periods=12)
model_fit2 = model2.fit(smoothing_level=0.2, smoothing_trend=0.2, smoothing_seasonal=0.2 , optimized=True)
forecast2 = model_fit2.forecast(len(test))

model = SimpleExpSmoothing(train['#Passengers'])
model_fit = model.fit()
forecast = model_fit.forecast(len(test))


plot_forecast(forecast , forecast1 , forecast2)



from statsmodels.tsa.holtwinters import SimpleExpSmoothing , Holt , ExponentialSmoothing

def plot_forecast(forecast2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'] , y = train['#Passengers'] , name = "train"))
    fig.add_trace(go.Scatter(x = test['Month'] , y = test['#Passengers'] , name = 'test'))
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast2 , name = "holt winters "))  
    fig.update_layout(
        width = 900
    )   
        
    return fig.show()   



model = ExponentialSmoothing(train['#Passengers'], trend='mul', seasonal='mul', seasonal_periods=12)
model_fit = model.fit(smoothing_level=0.2, smoothing_trend=0.2, smoothing_seasonal=0.2 , optimized=True)
forecast2 = model_fit.forecast(len(test))



plot_forecast(forecast2)
