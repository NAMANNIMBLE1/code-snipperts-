plt.rcParams['figure.figsize'] = (16, 9)
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Plotting the ACF
acf_plot = plot_acf(df['Passengers_seasonal_diff'][12:].dropna(), lags= 48) # lags is the number of lags to plot
plt.show()




plt.rcParams['figure.figsize'] = (16, 9)
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

plt.rcParams['figure.figsize'] = (16, 9)
# Plotting the PACF
pacf_plot = plot_pacf(df['#Passengers'].dropna(), lags= 48) # lags is the number of lags to plot
plt.show()

