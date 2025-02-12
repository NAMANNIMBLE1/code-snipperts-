train['naive_forecast'] = train['#Passengers'].iloc[-1]
test['naive_forecast'] = train['#Passengers'].iloc[-1]


# plot naive forecast
def plot_forecast(forecast , name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = train['Month'] , y = train['#Passengers'] , name = "train"))
    fig.add_trace(go.Scatter(x = test['Month'] , y = forecast , name = name))
    fig.show()  

plot_forecast(test['naive_forecast'] , 'naive forecast')    

