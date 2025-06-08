import matplotlib.pyplot as plt
from DataFrame import load_air_passengers
from Models import predict_ar_recursive

train, test = load_air_passengers(r"Classic Machin Learning\00 Time Serise\14 khordad 01 AR\AirPassnger.csv")

lags = 30
# استفاده از پیش‌بینی recursive (واقع‌گرایانه و عملی)
preds = predict_ar_recursive(train, test, lags=lags)

# برای رسم مقادیر پیش‌بینی شده باید ایندکس تست رو بهش بدی
plt.figure(figsize=(10,6))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Test')
plt.plot(test.index, preds, label='Predicted', linestyle='dashed', color='green')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('AR Model Prediction vs Actual (AirPassengers Dataset)')
plt.show()

# MAE
import numpy as np
mae = np.abs(np.array(preds) - test['value'].values).mean()
print(f"Mean Absolute Error (MAE): {mae:.2f}")
