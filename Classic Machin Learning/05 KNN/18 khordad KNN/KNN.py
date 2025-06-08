import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ۱. خواندن داده
df = pd.read_csv(r'Classic Machin Learning\05 KNN\18 khordad KNN\Mall_Customers.csv')

# ۲. پیش‌پردازش: تبدیل جنسیت به عددی (Label Encoding)
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])  # Female=0, Male=1

# ۳. انتخاب فیچرها و برچسب‌ها
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Gender_encoded']

# ۴. تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ۵. مدل KNN
k = 5  # تعداد همسایه
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ۶. پیش‌بینی روی داده تست
y_pred = knn.predict(X_test)

# ۷. ارزیابی مدل
print('دقت:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=le.classes_))

# ۸. نمایش نمونه داده تست و پیش‌بینی  
df_test = X_test.copy()
df_test['Real Gender'] = le.inverse_transform(y_test)
df_test['Predicted Gender'] = le.inverse_transform(y_pred)
print(df_test.head(10))
