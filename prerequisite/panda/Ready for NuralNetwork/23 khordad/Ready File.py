import pandas as pd

# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\panda\Ready for NuralNetwork\23 khordad\drug200.csv')
print('columns and firsts rows: ')
print(df.head())  # مشاهده چند سطر اول دیتا فریم
# حذف سطرهای با مقادیر گمشده
df.dropna(inplace=True)

# حذف ستون‌هایی که بیش از 50% مقادیرشان گمشده است
df.dropna(thresh=int(0.5*len(df)), axis=1, inplace=True)


# جایگزین با میانگین
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Na_to_K'].fillna(df['Na_to_K'].mean(), inplace=True)


# جایگزین با رایج‌ترین مقدار (مد)
df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].mode()[0], inplace=True)
df['BP'].fillna(df['BP'].mode()[0], inplace=True)
df['Drug'].fillna(df['Drug'].mode()[0], inplace=True)


# برای داده‌های عددی با توزیع غیر نرمال یا بیرون‌زدگی
df['Age'].fillna(df['Age'].median(), inplace=True)