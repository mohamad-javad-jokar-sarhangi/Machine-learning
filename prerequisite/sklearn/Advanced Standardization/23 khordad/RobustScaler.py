import pandas as pd
from sklearn.preprocessing import RobustScaler
# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\sklearn\Feature Encoding\23 khordad\drug200.csv')


# RobustScaler
scaler = RobustScaler()
df['Na_to_K_scaled'] = scaler.fit_transform(df[['Na_to_K']])