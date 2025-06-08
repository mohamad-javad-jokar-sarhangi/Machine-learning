import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# خواندن داده
df = pd.read_csv(r'Classic Machin Learning\07 Decision Tree\18 khordad Decision Tree Classification\Mall_Customers.csv')
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Gender_encoded']

# اسکیل داده اختیاری است اما معمولاً برای decision tree لازم نیست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ساخت، آموزش و پیش‌بینی مدل درخت تصمیم
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


# نمایش ساختار درخت
plt.figure(figsize=(20,10))
plot_tree(
    dtree,
    feature_names=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title('ساختار درخت تصمیم برای Gender')
plt.show()
