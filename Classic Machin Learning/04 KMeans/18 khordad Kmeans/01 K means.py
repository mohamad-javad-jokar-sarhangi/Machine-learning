import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ۱. خواندن فایل CSV
df = pd.read_csv(r'Classic Machin Learning\04 KMeans\18 khordad Kmeans\Mall_Customers.csv')
# پیش‌نمایش چند ردیف اول
print(df.head())

# ۲. انتخاب فیچرهای عددی مناسب برای خوشه‌بندی
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ۳. رسم داده اولیه (scatter plot)
plt.figure(figsize=(7,5))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], color='gray', s=80)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Distribution')
plt.grid(True)
plt.show()

# ۴. تعیین تعداد خوشه (K) و اجرای KMeans
k = 3  # فرضاً ۳ گروه می‌خوایم، می‌تونی تغییرش بدی!
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

df['Cluster'] = y_kmeans  # برچسب خوشه به dataframe اضافه میشه

# ۵. نمایش هر خوشه با رنگ متفاوت
colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
plt.figure(figsize=(8,6))
for i in range(k):
    plt.scatter(X[y_kmeans == i]['Annual Income (k$)'], 
                X[y_kmeans == i]['Spending Score (1-100)'],
                s=100, c=colors[i], label=f'Cluster {i+1}')
    
# مراکز خوشه
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='black', s=300, marker='X', label='Cluster Centers')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering of Customers')
plt.legend()
plt.grid(True)
plt.show()
