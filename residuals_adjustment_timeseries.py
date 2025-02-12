
# appending residuals and fitted values to the train dataframe 
train['fitted_values'] = model_fit.fittedvalues
train['residuals'] = model_fit.resid
print(train)


from statsmodels.graphics.tsaplots import plot_acf , plot_pacf

plot_pacf(train['residuals'] , lags= 30)
plot_acf(train['residuals'] , lags= 30)

from statsmodels.stats.diagnostic import acorr_ljungbox

# Apply Ljung-Box test on residuals
ljung_box_results = acorr_ljungbox(train['residuals'], return_df=True)
print(ljung_box_results)

# Extract p-values
p_values = ljung_box_results["lb_pvalue"]

# Check if any p-value is below 0.05 (significance level)
if (p_values < 0.05).any():
    print("Rejecting the null hypothesis (Residuals are autocorrelated)")
else:
    print("Accepting the null hypothesis (Residuals are independent)")


# histogram of residuals 

fig = px.histogram(train , x = 'residuals')
fig.update_layout(
    width = 900
)
fig.show()


plt.figure(figsize=(10 , 3))
sns.kdeplot(data = train , x = train['residuals'] , fill=True)

