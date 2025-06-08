import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# داده را بخوانید (مسیر را مطابق کامپیوتر خودت تنظیم کن)
df = pd.read_csv(r'Classic Machin Learning\01 Linear Regresion\18 khordad Normal\USA Housing Dataset.csv')

# فقط دو ویژگی اصلی را انتخاب می‌کنیم
X = df[['sqft_living', 'sqft_lot']]
y = df['price']

# داده‌ها را تقسیم می‌کنیم
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مدل را آموزش می‌دهیم
model = LinearRegression()
model.fit(X_train, y_train)

# داده تست را برای مقایسه نگه می‌داریم
X_test_np = X_test.values

# پیش‌بینی بر اساس داده تست
y_test_pred = model.predict(X_test)

# حالا صفحه مدل رو برای نمایش درست می‌کنیم:

# یک grid مناسب از مقادیر sqft_living و sqft_lot می‌سازیم
x1 = np.linspace(X['sqft_living'].min(), X['sqft_living'].max(), 20)
x2 = np.linspace(X['sqft_lot'].min(), X['sqft_lot'].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1, x2)
# مقدار price مدل روی این grid رو حساب می‌کنیم
y_grid = model.intercept_ + model.coef_[0]*x1_grid + model.coef_[1]*x2_grid

# نمودار سه‌بعدی
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
# داده واقعی (نقاط آبی)
ax.scatter(X_test['sqft_living'], X_test['sqft_lot'], y_test, color='blue', label='Actual Price')
# صفحه مدل رگرسیون (سطح قرمز)
ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.4, label='Regression Plane')

ax.set_xlabel('sqft_living (Metrage Living)')
ax.set_ylabel('Masahat (sqft_lot)')
ax.set_zlabel('Price (Price)')
plt.title('Linear Regression 3D - with 2 X')
plt.legend()
plt.show()
