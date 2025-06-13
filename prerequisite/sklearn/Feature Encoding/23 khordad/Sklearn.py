import pandas as pd
from sklearn.preprocessing import LabelEncoder

# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\sklearn\Feature Encoding\23 khordad\drug200.csv')
print('columns and firsts rows: ')
print(df.head())  # مشاهده چند سطر اول دیتا فریم



# label encoding
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

# one-hot encoding
dfn = pd.get_dummies(df, columns=['BP'])
print(dfn)

# Binning
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle', 'Old']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
print(df[['Age', 'Age_group']])