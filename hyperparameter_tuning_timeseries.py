# Import packages
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def hyperparameter_tuning_season_cv(n_splits: int,
                                    gammas: list[float],
                                    df: pd.DataFrame) -> pd.DataFrame:                                   
    """Function to carry out cross-validation hyperparameter tuning
    for the seasonal parameter in a Holt Winters' model. """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    error_list = []

    for gamma in gammas:
    
        errors = []
        
        for train_index, valid_index in tscv.split(df):
            train, valid = df.iloc[train_index], df.iloc[valid_index]
            
            model = ExponentialSmoothing(train['#Passengers'], trend='mul',
                                         seasonal='mul', seasonal_periods=12) \
                .fit(smoothing_seasonal=gamma)
                
            forecasts = model.forecast(len(valid))
            errors.append(mean_absolute_percentage_error(valid['#Passengers'], forecasts))

        error_list.append([gamma, sum(errors) / len(errors)])

    return pd.DataFrame(error_list, columns=['Gamma', 'MAPE'])
    
 
def plot_error_cv(df: pd.DataFrame,
                  title: str) -> None:                  
    """Bar chart to plot the errors from the different
    hyperparameters."""

    fig = px.bar(df, x='Gamma', y='MAPE')
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=800, title_x=0.5, height=400)

    return fig.show()
    
    

# Carry out cv for hyperparameter tuning for the seasonal parameter
error_df = hyperparameter_tuning_season_cv(df=df,
                                         n_splits=4,
                                         gammas=list(np.arange(0, 1.1, 0.1)))

# Plot the tuning results
plot_error_cv(df=error_df, title='Hyperparameter Results')



# simple terms 

tscv = TimeSeriesSplit(n_splits=4)
for train_index , valid_index in tscv.split(df):
    print("{}   {}".format(train_index , valid_index))

train , valid = df.iloc[train_index] , df.iloc[valid_index] 


from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train['#Passengers'], trend='mul', seasonal='mul', seasonal_periods=12)
model_fit = model.fit(smoothing_level=0.2, smoothing_trend=0.2, smoothing_seasonal=0.2 , optimized=True)
forecast = model_fit.forecast(len(valid)) 

# this before done

errors = mean_absolute_percentage_error(valid['#Passengers'] , forecast)
errors


