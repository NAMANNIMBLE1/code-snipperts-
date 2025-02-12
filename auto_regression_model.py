from scipy.stats import boxcox

df['Passengers_box_cox'] , lam = boxcox(df['#Passengers'])
df['Passengers_stationary']  = df['Passengers_box_cox'].diff()
df.dropna(inplace=True)  # stationary 

plot_timeline(df , x = df['Month'] , y = df['Passengers_stationary'], title = "stationary data")


from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(11,5))
plot_pacf(df['Passengers_stationary'] )

from statsmodels.tsa.ar_model import AutoReg , ar_select_order


# train and test split
train = df.iloc[: -int(len(df) * 0.2)]
test = df.iloc[-int(len(df) * 0.2):]


selector = ar_select_order(train['Passengers_stationary'] , 15)
model = AutoReg(train['Passengers_stationary'] , lags= selector.ar_lags).fit()
transformed_forecast = model.forecast(steps=len(test))
print(transformed_forecast)



# vizualization and plotting 

# Import packages
from scipy.special import inv_boxcox
import plotly.graph_objects as go

# Get forecasts and convert to actual passenger volumes
transformed_forecasts = list(model.forecast(steps=len(test)))
boxcox_forecasts = []
for idx in range(len(test)):
    if idx == 0:
        boxcox_forecast = transformed_forecasts[idx] + train['Passengers_box_cox'].iloc[-1]
    else:
        boxcox_forecast = transformed_forecasts[idx] + boxcox_forecasts[idx-1]

    boxcox_forecasts.append(boxcox_forecast)

forecasts = inv_boxcox(boxcox_forecasts, lam)


def plot_forecasts(forecasts: list[float],
                   title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Month'], y=train['#Passengers'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Month'], y=test['#Passengers'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Month'], y=forecasts, name='Forecast'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=650, title_x=0.5, height=400, xaxis_title='Date',
                      yaxis_title='Passenger Volume')

    return fig.show()


# Plot the forecasts
plot_forecasts(forecasts, 'Autoregression')
