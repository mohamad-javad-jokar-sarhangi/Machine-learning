from DataFrame import load_data
from Models import fit_prophet
import pandas as pd
import matplotlib.pyplot as plt

df = load_data(r"Classic Machin Learning\00 Time Serise\14 khordad 05 SARIMA\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
df_prophet = df.rename(columns={'date':'ds', 'close':'y'})[['ds','y']]

# Train/Test split (Prophet کل سری رو model میکنه و future رو پیش‌بینی میکنه)
periods = 30  # مثلا 30 روز آینده

model, forecast = fit_prophet(df_prophet, periods=periods, freq='M')

plt.figure(figsize=(14,5))
plt.plot(df_prophet['ds'], df_prophet['y'], label='True Data')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Prediction', color='purple')
plt.title('Prophet Model')
plt.legend()
plt.show()

# اگر بخواهی فقط روی دیتای test خطا حساب کنی:
test = df_prophet.iloc[-periods:]
pred = forecast.iloc[-periods:]["yhat"].values
mae = (test['y'].values - pred).mean()
print("MAE:", mae)