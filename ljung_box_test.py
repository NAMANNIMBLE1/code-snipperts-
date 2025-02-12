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
