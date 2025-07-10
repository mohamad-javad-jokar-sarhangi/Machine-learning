import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt

# ======== 1. بارگذاری و پیش‌پردازش داده‌ها ========

# فرض: نام فایل شما "gold_15m.csv"
df = pd.read_csv(
    r'Deep Learning\02 CNN\16 tir Level 01\XAUUSD M1 2025-01-06 to 2025-04-17.csv',
    sep='\t',
    header=None,
    names=['Date', 'Open', 'High', 'Low', 'Close', 'Time']
)

# اگر اسم ستون‌ها مثلا دیت‌تایم، اوپن، های، لو، کلوز هستن
# و ستون آخر فقط مقدار 15 هست:
df = df.drop(df.columns[-1], axis=1)  # حذف ستون آخر (15 ثابت)

# اگر ستون «تاریخ و ساعت» داری، بهتره پاک شه چون تو شبکه CNN فقط داده عددی می‌دیم:
df = df.drop(df.columns[0], axis=1)   # حذف ستون تاریخ/ساعت

# حالا شما داری:  [Open, High, Low, Close]

# ======== 2. نرمال‌سازی کل داده =========

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# ======== 3. ساخت دیتاست CNN با window ========
window_size = 30   # ۳۰ کندل (۷.۵ ساعت)
X = []
y = []

for i in range(window_size, len(data_scaled)):
    X.append(data_scaled[i-window_size:i, :])      # کل ویژگی‌ها!
    y.append(data_scaled[i, 3])                    # کلوزِ کندل بعدی (index=3)
    
X, y = np.array(X), np.array(y)

# ======== 4. تقسیم آموزش/تست (۸۰/۲۰ یا هرچقدر دوست داری) ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)   # shuffle=False چون تایم‌سریزه

# ======== 5. ساخت مدل CNN =========
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))  # پیش‌بینی قیمت کلوز بعدی

model.compile(optimizer='adam', loss='mse')
model.summary()

# ======== 6. آموزش مدل =========
history = model.fit(
    X_train, y_train,
    epochs=20,              # هرچقدر صلاح دونستی زیاد/کم کن
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ======== 7. ارزیابی مدل =========
y_pred = model.predict(X_test)
# معکوس مقیاس‌بندی برای y و y_pred (فقط روی ستون کلوز!)
close_index = 3
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)], axis=1)
)[:, close_index]
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([np.zeros((len(y_pred), 3)), y_pred], axis=1)
)[:, close_index]

# ======== 8. رسم نتایج =========
plt.figure(figsize=(15,6))
plt.plot(y_test_rescaled, label='Real(Close)')
plt.plot(y_pred_rescaled, label='Perdiction (Close)')
plt.legend()
plt.title('Gold Price Close Prediction (Test Set)')
plt.xlabel('test or time')
plt.ylabel('Price')
plt.show()
