import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- خواندن داده
df = pd.read_csv(r'Classic Machin Learning\02 Logistic Regression\18 khordad Normal\Social_Network_Ads.csv')
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0

# --- فقط دو ویژگی برای تجسم دو‌بعدی
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# --- تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- آموزش مدل
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- ارزیابی
print("===== 2D Version =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- رسم مرز تصمیم
x_min, x_max = X['Age'].min()-1, X['Age'].max()+1
y_min, y_max = X['EstimatedSalary'].min()-5000, X['EstimatedSalary'].max()+5000
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X_test['Age'], X_test['EstimatedSalary'], c=y_test, edgecolor='k', cmap='bwr', s=70)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Logistic Regression Decision Boundary (2 features)')
plt.show()
