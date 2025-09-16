import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\sept month\15th_cluster intro\mall customers segmentation\data\Mall_Customers.csv")

x = dataset.iloc[:,[3,4]].values
#clusters doesnt required y dependent variable
from sklearn.cluster import KMeans

wcss = [] # to create a elbow graph

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSs')
plt.show()# to get to know how many clusters we can create by seeing the graph

#training the k-means model on dataset
kmeans = KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualisation of the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# cluster 1 - average
# cluster 2 - vip
# clsuter 3 - clssic
# cluster 4 - less pruchase custo
# cluster 5 - less income high spending

dataset['cluster'] = y_kmeans
dataset # to print the predictions with new column in dataset










