from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA

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

