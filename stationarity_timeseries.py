#third way taking diff and log to make it stationary
# second way 
df['Passengers_log'] = np.log(df['Passengers'])
df.head()
df['Passengers_log_diff'] = df['Passengers_log'].diff()
df.head()

import plotly.express as px

def plot_timeline(df, x, y, title): 
    fig = px.line(df, x=x, y=y, title=title)  # Ensure x and y are keyword arguments
    fig.show()

plot_timeline(df, x="Month", y="Passengers_log_diff", title="Air Passengers Timeline")  

from statsmodels.tsa.stattools import adfuller
# checking for stationarity using adfuller
result = adfuller(df['Passengers_log_diff'].dropna())

print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# Interpretation
if result[1] < 0.05:
    print("The time series is stationary (Reject H0).")
else:
    print("The time series is non-stationary (Fail to reject H0).")

# other methods are box cox and taking diffrene 
