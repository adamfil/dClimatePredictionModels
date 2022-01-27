import requests
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import numpy as np
    
def temp_prediction(days_in_future: int, lat: int, long: int):
    #fetch data 
    dataset = 'prismc-tmax-daily'
    my_token = 'REDACTED'

    my_url = 'https://api.dclimate.net/apiv3/grid-history/' + dataset + '/' + str(lat) + '_' + str(long)
    head = {"Authorization": my_token}
    r = requests.get(my_url, headers=head)
    data = r.json()["data"]
    index = pd.to_datetime(list(data.keys()))
    values = [float(s.split()[0]) if s else None for s in data.values()]
    series = pd.Series(values, index=index)
    df = series.to_frame(name='Value')
    df = df[~df.index.astype(str).str.contains('02-29')]

    #algorithm 
    hw_model = ExponentialSmoothing(df["Value"],
                              trend    ="add",
                              seasonal = "add", 
                              seasonal_periods=365, 
                              damped=False).fit(use_boxcox='log')

    hw_fitted = hw_model.fittedvalues

    hw_resid = hw_model.resid

    #Adding the mean of the residuals to correct the bias.
    py_hw = hw_model.forecast(days_in_future) + np.mean(hw_resid)

    #output
    return(py_hw[-1])
