import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# فرض: فایل داده را ذخیره کردی با نام مثلاً applications.csv
df = pd.read_csv(r'Classic Machin Learning\07 Decision Tree\18 khordad Decision Tree Regression\MBA.csv')

# حذف ستون id
df = df.drop('application_id', axis=1)

# حذف ردیف‌هایی که مقدار هدف‌شان خالی است ("gmat" خالی)
df = df[df['gmat'].notnull()]

# تبدیل categorical variables به numeric (Label Encoder)
label_columns = ['gender', 'major', 'race', 'work_industry', 'admission']
for col in label_columns:
    df[col] = df[col].fillna('Unknown')
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# تبدیل باینری (True/False) به int
df['international'] = df['international'].astype(int)

# حذف یا مقداردهی work_exp خالی
df['work_exp'] = df['work_exp'].fillna(df['work_exp'].median())

# همین کار را برای gpa هم بکن اگر خالی دارد:
df['gpa'] = df['gpa'].fillna(df['gpa'].median())

X = df.drop('gmat', axis=1)
y = df['gmat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



plt.figure(figsize=(20,10))
plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True)
plt.title('ساختار درخت تصمیم برای پیش‌بینی GMAT')
plt.show()
