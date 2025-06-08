import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# تولید داده‌های تصادفی با استفاده از make_blobs
data, _ = make_blobs(n_samples=200, centers=4, random_state=42, cluster_std=50)

print("Generated Customers Data:")
print(data)

# ساخت مدل K-Means با K=3 (سه خوشه)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)  # تعداد خوشه‌ها: 3

# اجرا و پیداکردن خوشه‌ها
kmeans.fit(data)
clusters = kmeans.predict(data)  # پیش‌بینی خوشه‌ها برای داده‌ها

print("Clustering of Customers:")
print(clusters)

# مراکز خوشه‌ها (centroids)
print("centroids:")
print(kmeans.cluster_centers_)

# داده‌های خوشه‌بندی شده
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', label='Customers')

# نمایش مراکز خوشه‌ها
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', label='centroids of clusters')

plt.xlabel('Total Purchases')
plt.ylabel('Number of Purchases')
plt.legend()
plt.title('Customer Segmentation with K-Means')
plt.savefig('customer_segmentation.png')
print("The customer_segmentation.png file has been saved.")
plt.show()
