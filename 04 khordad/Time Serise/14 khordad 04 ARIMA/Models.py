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

def fit_arma(series, p=1, q=1):
    """
    Fit ARMA(p,q) model to the given series.
    Args:
        series: pd.Series, target time series
        p: int, number of AR lags
        q: int, number of MA lags
    Returns:
        Fitted statsmodels model object
    """
    # ARMA(p, q) is ARIMA(p, 0, q)
    model = ARIMA(series, order=(p, 0, q))
    model_fit = model.fit()
    return model_fit



def fit_arima(series, p=1, d=1, q=1):
    """
    Fit ARIMA(p,d,q) model to the given series.
    Args:
        series: pd.Series, target time series
        p: int, AR lags
        d: int, differencing order
        q: int, MA lags
    Returns:
        Fitted statsmodels ARIMA model object
    """
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

