import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- داده‌خوانی و پیش‌پردازش
df = pd.read_csv(r'Classic Machin Learning\02 Logistic Regression\18 khordad Normal\Social_Network_Ads.csv')
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0

# --- سه ویژگی
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- مدل لجستیک و آموزش
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("===== 3D Version =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- رسم سه‌بعدی (برای gender=1 (male) یا gender=0 (female)، یکی را انتخاب کن!)
gender_val = 1  # 1: male, 0: female
# مش بندی در دو ویژگی Age و Salary برای مقدار ثابت Gender
age_range = np.arange(X['Age'].min(), X['Age'].max(), 1)
salary_range = np.arange(X['EstimatedSalary'].min(), X['EstimatedSalary'].max(), 1000)
age_grid, salary_grid = np.meshgrid(age_range, salary_range)
gender_grid = np.full_like(age_grid, gender_val)

# ترکیب مش بندی برای پیش‌بینی
mesh_points = np.c_[gender_grid.ravel(), age_grid.ravel(), salary_grid.ravel()]
Z = model.predict(mesh_points)
Z = Z.reshape(age_grid.shape)

# رسم
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
# نمایش داده واقعی (فقط همون جنسیت ثابت شده)
idx = X_test['Gender'] == gender_val
ax.scatter(X_test.loc[idx, 'Age'],
           X_test.loc[idx, 'EstimatedSalary'],
           y_test[idx],
           c=y_test[idx], cmap='bwr', s=50, label='Data')

# رسم سطح مرز تصمیم
ax.plot_surface(age_grid, salary_grid, Z, alpha=0.3, cmap='bwr')

ax.set_xlabel('Age')
ax.set_ylabel('Estimated Salary')
ax.set_zlabel('Purchased')
plt.title(f'Logistic Regression Decision Surface (Gender={"Male" if gender_val==1 else "Female"})')
plt.show()
