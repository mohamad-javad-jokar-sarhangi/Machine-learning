import pandas as pd
from sklearn.decomposition import PCA
# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\sklearn\Feature Encoding\23 khordad\drug200.csv')

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df[['Na_to_K', 'Age', 'Cholesterol']] )