from DataFrame import load_data
from Models import fit_ma
import matplotlib.pyplot as plt

df = load_data(r"04 khordad\Time Serise\14 khordad AR\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
series = df['close']

# Fit MA model
model_fit = fit_ma(series, q=3)  # مقدار q را می‌توان تست کرد
pred = model_fit.predict(start=len(series)//2, end=len(series)-1)

# Plot
plt.plot(series.values, label='True Data')
plt.plot(range(len(series)//2, len(series)), pred, label='MA Prediction')
plt.legend()
plt.show()

# Print summary
print(model_fit.summary())