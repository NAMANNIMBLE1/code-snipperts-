# average forecast
def average_forecast(forecast , name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'] , y = train['#Passengers'] , name = "train"))
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast , name = name))
    fig.show()  


test['average_forecast'] = train['#Passengers'].mean()
average_forecast(test['average_forecast'] , 'average forecast')    

