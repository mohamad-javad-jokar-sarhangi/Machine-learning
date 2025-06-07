from DataFrame import load_data
from Models import fit_sarima
import matplotlib.pyplot as plt

df = load_data(r"04 khordad\Time Serise\14 khordad 05 SARIMA\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
series = df['close']

train_size = int(0.7 * len(series))
train, test = series[:train_size], series[train_size:]

order = (2,1,2)
seasonal_order = (1,1,1,12)  # برای ماهانه s=12. برای داده طلا ممکن است فصل خاصی نباشد!

model_fit = fit_sarima(train, order=order, seasonal_order=seasonal_order)
pred = model_fit.predict(start=test.index[0], end=test.index[-1])

plt.figure(figsize=(14,5))
plt.plot(series, label='True Data')
plt.plot(test.index, pred, label='SARIMA Prediction', color='green')
plt.title('SARIMA Model')
plt.legend()
plt.show()

mae = (pred - test).abs().mean()
print("MAE:", mae)