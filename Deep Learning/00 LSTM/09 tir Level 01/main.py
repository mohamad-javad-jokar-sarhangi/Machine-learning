### Part 1 Ready Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
### Part 2 Create LSTM with Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
### Part 3 Show Result
import matplotlib.pyplot as plt

### Part 4 Stop Change Model
import tensorflow as tf
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


### Part 1 Ready Data
# --- 1. بارگذاری دیتا و تبدیل تاریخ ---
df = pd.read_csv(
    r'Deep Learning\00 LSTM\09 tir Level 01\XAUUSD D1 2008-08-08 to 2025-04-18.csv',
    sep='	',
    header=None,
    names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# فقط ستون‌های مورد نیاز (می‌تونی فیچر اضافه کنی)
feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[feat_cols].copy()

# --- 2. نرمال‌سازی ---
scaler = MinMaxScaler()
data[feat_cols] = scaler.fit_transform(data[feat_cols])

# --- 3. ساخت X و y به صورت پنجره‌ای ---
window = 5

X, y = [], []
for i in range(window, len(data)):
    X.append(data.iloc[i-window:i].values)   # سکانس روزهای گذشته
    y.append(data.iloc[i]['Close'])          # فقط Close روز آینده

X, y = np.array(X), np.array(y)    # X.shape=(نمونه, window, ویژگی)

# --- 4. تقسیم بر اساس زمان (نه تصادفی) ---
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

### Part 2 Create LSTM with Keras
# --- 1. تعریف مدل ---
model = Sequential()
model.add(LSTM(64, input_shape=(window, len(feat_cols)), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # خروجی عددی

model.compile(loss='mse', optimizer='adam')

# --- 2. آموزش مدل ---
history = model.fit(X_train, y_train, epochs=35, batch_size=32, validation_split=0.1)

# --- 3. ارزیابی ---
loss = model.evaluate(X_test, y_test)
print('Test MSE:', loss)

# --- 4. پیش‌بینی و بازگرداندن قیمت به حالت اصلی ---
y_pred = model.predict(X_test)

# بازیابی اسکیل برای مشاهده عدد واقعی
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_test), 4)), y_test.reshape(-1,1)), axis=1)
)[:,-1]
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_pred), 4)), y_pred), axis=1)
)[:,-1]

### Part 3 Show Result
plt.figure(figsize=(12,5))
plt.plot(y_test_rescaled, label='Real', color='red')
plt.plot(y_pred_rescaled, label='Perdiction', color='blue')
plt.legend()
plt.title('Perdiction Close with LSTM')
plt.xlabel('Sample')
plt.ylabel('Close')
plt.show()