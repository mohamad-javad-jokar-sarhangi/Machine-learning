from prophet import Prophet
import pandas as pd

def fit_prophet(df, periods=30, freq='D'):
    """
    Fit Facebook Prophet model.
    df: DataFrame with columns ['ds', 'y']
    periods: Number of future periods to forecast (e.g. 30 days)
    freq: Frequency string (e.g. 'D' for daily)
    Returns:
        (model, forecast DataFrame)
    """
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast
