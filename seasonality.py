# taking a season diff and plotting
df['Passengers_seasonal_diff'] = df['#Passengers'].diff(12) # 6 is the period of the seasonality
plot_timeline(df, x="Month", y="Passengers_seasonal_diff", title="Air Passengers Seasonal Diff")


