import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# مرحله ۱: خواندن داده
df = pd.read_csv(r'Classic Machin Learning\01 Linear Regresion\18 khordad\USA Housing Dataset.csv')

# مرحله ۲: انتخاب یک ویژگی (مثلاً sqft_living)
X = df[['sqft_living']]  # نیاز به دو بعدی بودن ورودی برای sklearn
y = df['price']

# مرحله ۳: ساخت و آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X, y)

# مرحله ۴: پیش‌بینی بر اساس داده‌های موجود (برای رسم خط مدل)
y_pred = model.predict(X)

# مرحله ۵: رسم داده‌ها و خط رگرسیون
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Real Data', alpha=0.3)
plt.plot(X, y_pred, color='red', linewidth=2, label='Model Prediction')
plt.xlabel('Meterage Home(sqft_living)')
plt.ylabel('Home Price')
plt.title('Home Price vs Meterage (sqft_living)')
plt.legend()
plt.grid(True)
plt.show()
