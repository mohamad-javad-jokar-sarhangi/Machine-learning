from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def fit_ma(series, q=1):
    # MA(q) is ARIMA(0,0,q)
    model = ARIMA(series, order=(0, 0, q))
    model_fit = model.fit()
    return model_fit
