### Part 1 Loading Data
import pandas as pd
### Part 2 sliding window
import numpy as np



### Part 1 Loading Data
# خواندن فایل
df = pd.read_csv(r'Classic Machin Learning\08 Windowing\09 tir\XAUUSD D1 2008-08-08 to 2025-04-18.csv', sep='\t', header=None,
                 names=["Date", "Open", "High", "Low", "Close", "Volume"])

# مطمئن شدن که مرتب بر اساس تاریخ هستیم
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df.head()

### Part 2 sliding window
window = 5  # تعداد روزهای ورودی مدل

X = []
y = []

for i in range(window, len(df)):
    features = df.loc[i - window:i - 1, ['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
    X.append(features)
    y.append(df.loc[i, 'Close'])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # تعداد نمونه، تعداد ویژگی (5روز*5ویژگی=25)
print("y shape:", y.shape)