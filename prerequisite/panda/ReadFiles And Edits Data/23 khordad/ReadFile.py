import pandas as pd

# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\panda\ReadFiles And Edits Data\23 khordad\drug200.csv')
print('columns and firsts rows: ')
print(df.head())  # مشاهده چند سطر اول دیتا فریم
print('____________________________________________________')
print('Columns: ')
print(df.columns) #نام ستون ها
print('____________________________________________________')
print('Data types: ')
print(df.dtypes) # نوع داده ها
print('____________________________________________________')
df.columns = ['سن', 'جنس', 'فشار', 'کلسترول', 'نمک-پتاسیم', 'دارو']
print('Columns: ')
print(df.columns) #نام ستون ها
df.to_csv(r'prerequisite\panda\ReadFiles And Edits Data\23 khordad\New_drug200.csv', index=False)
print('____________________________________________________')