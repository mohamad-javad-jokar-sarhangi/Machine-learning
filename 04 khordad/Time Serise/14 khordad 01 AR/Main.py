from DataFrame import load_data, train_test_split
from Models import train_ar, forecast_ar
import matplotlib.pyplot as plt

df = load_data(r"04 khordad\Time Serise\14 khordad 01 AR\XAUUSD D1 2008-08-08 to 2025-04-18.csv")
train, test = train_test_split(df, split_ratio=0.8)

ar_model = train_ar(train['close'], lags=1000)
steps = len(test)
preds = forecast_ar(ar_model, steps=steps)

plt.plot(list(train['date']) + list(test['date']),
         list(train['close']) + list(test['close']), label='True Data')
plt.plot(test['date'], preds, label='AR Prediction')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AR Model for Gold Close Price')
plt.show()
