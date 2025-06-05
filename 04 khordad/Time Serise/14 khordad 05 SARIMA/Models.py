from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Fit Seasonal ARIMA (SARIMA) model.
    series: pd.Series, target time series
    order: (p,d,q)
    seasonal_order: (P,D,Q,s), e.g. (1,1,1,12) for monthly
    """
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit
