import pandas as pd
from sklearn.manifold import TSNE
# خواندن فایل CSV
df = pd.read_csv(r'prerequisite\sklearn\Feature Encoding\23 khordad\drug200.csv')

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(df[['Na_to_K', 'Age', 'Cholesterol']])