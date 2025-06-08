from DataFrame import load_data
from Models import fit_arma
import matplotlib.pyplot as plt

df = load_data(r"Classic Machin Learning\00 Time Serise\14 khordad 03 ARMA\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
series = df['close']

train_size = int(0.7 * len(series))  # یا درصد دلخواه
train, test = series[:train_size], series[train_size:]

# مدل ARMA با پارامترهای پیشنهادی
p = 2  # تعداد lag برای AR
q = 2  # تعداد lag برای MA

model_fit = fit_arma(train, p=p, q=q)

# پیش‌بینی روی دیتای تست
pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# رسم نمودار
plt.figure(figsize=(14,5))
plt.plot(series, label='True Data')
plt.plot(test.index, pred, label=f'ARMA({p},{q}) Prediction', color='crimson')
plt.title('ARMA Model for Gold Close Price')
plt.legend()
plt.show()

# MAE محاسبه
mae = (pred - test).abs().mean()
print("MAE:", mae)