from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def train_ar(series, lags=3):
    model = AutoReg(series, lags=lags)
    model_fit = model.fit()
    return model_fit

def forecast_ar(model_fit, steps):
    preds = model_fit.predict(start=len(model_fit.model.endog),
                             end=len(model_fit.model.endog)+steps-1)
    return preds


def fit_ma(series, q=1):
    # MA(q) is ARIMA(0,0,q)
    model = ARIMA(series, order=(0, 0, q))
    model_fit = model.fit()
    return model_fit
