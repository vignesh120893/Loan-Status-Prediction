import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'path_to_your/Credit Card_Clustering.csv'
credit_card_data = pd.read_csv(file_path)

# Display the first few rows and basic information
print("Dataset Head:\n", credit_card_data.head())
print("\nDataset Info:\n", credit_card_data.info())
print("\nDataset Description:\n", credit_card_data.describe())

# Handle missing values
credit_card_data.fillna(credit_card_data.mean(), inplace=True)

# Drop the CUST_ID column for clustering
credit_card_data.drop('CUST_ID', axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
credit_card_data_scaled = scaler.fit_transform(credit_card_data)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(credit_card_data_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Based on the elbow plot, choose an appropriate number of clusters (e.g., 3 or 4)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(credit_card_data_scaled)

# Add the cluster labels to the original data
credit_card_data['Cluster'] = kmeans_labels

# Evaluate clustering performance
silhouette_avg = silhouette_score(credit_card_data_scaled, kmeans_labels)
print(f'Silhouette Score: {silhouette_avg}')

# Analyze the clusters
cluster_analysis = credit_card_data.groupby('Cluster').mean()
print("\nCluster Analysis:\n", cluster_analysis)

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=credit_card_data_scaled[:, 0], y=credit_card_data_scaled[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('Cluster Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Save the clustered data
credit_card_data.to_csv('path_to_your/clustered_credit_card_data.csv', index=False)
