from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

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

