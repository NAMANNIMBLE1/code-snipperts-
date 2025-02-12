from statsmodels.tsa.stattools import adfuller

# Example time series data
import numpy as np
import pandas as pd

# Creating a sample non-stationary time series (random walk)
np.random.seed(42)
data = np.cumsum(np.random.randn(100))  # Random walk series

# Perform ADF test
result = adfuller(data)

# Print the results
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# Interpretation
if result[1] < 0.05:
    print("The time series is stationary (Reject H0).")
else:
    print("The time series is non-stationary (Fail to reject H0).")
