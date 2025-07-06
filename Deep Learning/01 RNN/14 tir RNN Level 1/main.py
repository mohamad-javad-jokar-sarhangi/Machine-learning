import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# 1. بارگذاری داده
df = pd.read_csv(
    r'Deep Learning\01 RNN\14 tir RNN Level 1\XAUUSD M1 2025-01-06 to 2025-04-17.csv',
    sep='\t',
    header=None,
    names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
)

# 2. فقط ستون Close رو نگه می‌داریم
data = df[['Close']]

# 3. نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# 4. ایجاد توالی‌های زمانی (به عنوان مثال با window=30)
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window = 30  # تعداد روزهایی که برای هر ورودی در نظر می‌گیریم
X, y = create_sequences(data['Close'].values, window)

# 5. تبدیل داده‌ها به شکل سه‌بعدی: [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 6. تقسیم دیتاست به train و test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # برای داده‌ی زمانی، shuffle نکن!
)

# 7. ساخت مدل RNN
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(window, 1)),
    Dense(1)
])

# 8. کامپایل مدل
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# 9. آموزش مدل
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 10. پیش‌بینی
pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
pred_inv = scaler.inverse_transform(pred)+3

# 11. مصورسازی
plt.figure(figsize=(14,6))
plt.plot(y_test_inv, label='Real Close', color='blue')
plt.plot(pred_inv, label='Predicted Close', color='orange')
plt.title('Gold Price Prediction with SimpleRNN')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.tight_layout()
plt.show()

# نمایش چند نمونه مقایسه‌ای
for i in range(5):
    print(f'Actual: {y_test_inv[i][0]:.2f}, Predicted: {pred_inv[i][0]:.2f}')
