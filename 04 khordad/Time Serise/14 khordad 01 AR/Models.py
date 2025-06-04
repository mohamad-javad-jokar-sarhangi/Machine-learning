from statsmodels.tsa.ar_model import AutoReg

def fit_ar(train, lags=3):
    model = AutoReg(train['value'], lags=lags)
    model_fit = model.fit()
    return model_fit

def predict_ar_recursive(train, test, lags=3):
    """
    پیش‌بینی قدم به قدم برای داده تست (بدون استفاده از مقادیر واقعی تست).
    """
    history = list(train['value'])
    preds = []
    for t in range(len(test)):
        model = AutoReg(history, lags=lags)
        model_fit = model.fit()
        yhat = model_fit.predict(start=len(history), end=len(history))
        preds.append(yhat[0])
        history.append(yhat[0])
    return preds
