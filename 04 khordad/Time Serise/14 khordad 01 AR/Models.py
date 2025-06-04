from statsmodels.tsa.ar_model import AutoReg

def train_ar(series, lags=3):
    model = AutoReg(series, lags=lags)
    model_fit = model.fit()
    return model_fit

def forecast_ar(model_fit, steps):
    preds = model_fit.predict(start=len(model_fit.model.endog),
                             end=len(model_fit.model.endog)+steps-1)
    return preds
