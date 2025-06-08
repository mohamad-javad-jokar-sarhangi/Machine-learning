from DataFrame import load_data
from Models import fit_arima
import matplotlib.pyplot as plt

df = load_data(r"Classic Machin Learning\00 Time Serise\14 khordad 04 ARIMA\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
series = df['close']

train_size = int(0.7 * len(series))
train, test = series[:train_size], series[train_size:]

# پارامترهای ARIMA: پیشنهاد اولیه (p=2, d=1, q=2)
p, d, q = 2, 1, 2

model_fit = fit_arima(train, p=p, d=d, q=q)

# پیش‌بینی روی بازه تست
pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# رسم نتایج پیش‌بینی vs داده واقعی
plt.figure(figsize=(14,5))
plt.plot(series, label='True Data')
plt.plot(test.index, pred, label=f'ARIMA({p},{d},{q}) Prediction', color='teal')
plt.title('ARIMA Model for Gold Close Price')
plt.legend()
plt.show()

# محاسبه MAE (خطای میانگین مطلق)
mae = (pred - test).abs().mean()
print("MAE:", mae)