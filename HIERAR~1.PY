#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\19th,20th\3.HIERARCHICAL CLUSTERING\Mall_Customers.csv")


#Removing the unnecessary attributes from the dataset to form the I.V as here for clustering algorithms, there is no D.V
X = dataset.iloc[:, [3, 4]].values


#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#scipy is an open source python libray which contain tools to do clustering and build the dendrogram
#We are not going to import whole scipy, but we are importing only scipy which related to the cluster and hierarchy


dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
#We are going to build the dendrogram with only one line of code 
#linkage is one of the hierarchical clustering algorithm & you have to build the linkage on X 
#ward method actually try to minimise the variance on each cluster & in k-means we minimise the sum of squared 
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.show()


#Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)


#Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


#To find out which customer belongs to which cluster
dataset_test = dataset
dataset_test["Cluster No.(hc)"] = pd.DataFrame(y_hc+1)
dataset_test.to_csv("New Mall Customers_hc.csv")


#To find out the path location where exactly the new .csv file saved in our machine
import os
os.getcwd()
