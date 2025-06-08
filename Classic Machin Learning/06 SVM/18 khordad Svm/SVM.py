import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. خواندن و آماده‌سازی داده
df = pd.read_csv(r'Classic Machin Learning\06 SVM\18 khordad Svm\Mall_Customers.csv')
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])   # Male=1, Female=0

# 2. انتخاب فیچرها و برچسب
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Gender_encoded']

# 3. اسکیل کردن فیچرها (برای SVM مهم است)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. تقسیم به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. ساخت مدل SVM و آموزش
svm = SVC(kernel='linear', C=1, random_state=42)   # کرنل خطی (برای شروع)، می‌تونی 'rbf' یا 'poly' هم تست کنی
svm.fit(X_train, y_train)

# 6. پیش‌بینی و ارزیابی
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 7. نمایش پیش‌بینی چند نمونه
df_result = pd.DataFrame(X_test, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
df_result['Real Gender'] = le.inverse_transform(y_test)
df_result['Predicted Gender'] = le.inverse_transform(y_pred)
print(df_result.head(10))
