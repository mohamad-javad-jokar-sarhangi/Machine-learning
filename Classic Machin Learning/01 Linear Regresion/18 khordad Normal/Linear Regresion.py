import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# مرحله ۱: خواندن داده
df = pd.read_csv(r'Classic Machin Learning\01 Linear Regresion\18 khordad Normal\USA Housing Dataset.csv')

# مرحله ۲: انتخاب یک ویژگی (مثلاً sqft_living)
X = df[['sqft_living']]  # نیاز به دو بعدی بودن ورودی برای sklearn
y = df['price']
print(df[['sqft_living', 'sqft_lot', 'price']].head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مرحله ۳: ساخت و آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X, y)

# مرحله ۴: پیش‌بینی بر اساس داده‌های موجود (برای رسم خط مدل)
y_test_pred = model.predict(X_test)

#  نمایش ضرایب (وزن‌ها)
print("Coef:", model.coef_)   # ضرایب هر ویژگی
print("Intercept:", model.intercept_)  # عدد ثابت (bias)

# مرحله ۵: رسم داده‌ها و خط رگرسیون
plt.scatter(X_test['sqft_living'], y_test, color='black', label='Real')
plt.plot(X_test['sqft_living'], y_test_pred, color='red', label='Regression Line')
plt.xlabel('Metres')
plt.ylabel('Price')
plt.legend()
plt.show()

