# importing library
from statsmodels.tsa.seasonal import seasonal_decompose

decomposed = seasonal_decompose(df['Passengers_log_diff'].dropna(), model='additive', period=12)  # Assuming monthly data
decomposed.plot()
plt.show()


decomposed = seasonal_decompose(df['Passengers'].dropna(), model='multiplicative', period=12)  # Assuming monthly data
decomposed.plot()
plt.show()
