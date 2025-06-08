import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- ۱. خواندن داده
df = pd.read_csv(r'Classic Machin Learning\03 PCA\18 khordad PCA\framingham.csv')

# --- ۲. حذف سطرهای ناقص (خیلی مهم!)
df = df.dropna(axis=0)

# --- ۳. جدا کردن ورودی‌ها و خروجی
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# --- ۴. استانداردسازی فیچرها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ۵. اجرای PCA (اجرا برای n مؤلفه=تعداد فیچرها)
pca = PCA()
X_pca_all = pca.fit_transform(X_scaled)

# -- نمایش نسبت واریانس مؤلفه‌ها (چندتا مؤلفه کافی است؟)
explained = pca.explained_variance_ratio_
print(f'نسبت واریانس توضیح داده‌شده هر مؤلفه:\n{explained}')  
print(f'واریانس تجمعی با اولین ۲ مؤلفه:', explained[:2].sum())
print(f'واریانس تجمعی با اولین ۳ مؤلفه:', explained[:3].sum())
print(f'واریانس تجمعی با اولین ۴ مؤلفه:', explained[:4].sum())
print(f'واریانس تجمعی کل:', explained.sum())

# -- رسم SCREE Plot برای انتخاب n_components مناسب
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(explained), marker='o')
plt.xlabel('X = تعداد مؤلفه')
plt.ylabel('Total Variance')
plt.title('Variance Explained by Principal Components')
plt.grid(True)
plt.show()

# --- ۶. کاهش بعد داده به ۲ مؤلفه و نمایش تصویری
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='bwr', s=30, edgecolor='k', alpha=0.8)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Framingham Dataset (PCA)')
plt.colorbar(label='TenYearCHD')
plt.grid(True)
plt.show()
