#!/usr/bin/env python
# coding: utf-8

# # Case Study: Fast Food

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('mcdonalds.csv')

# Extract the relevant columns for segmentation
segmentation_data = data.iloc[:, 0:11]


# In[ ]:


segmentation_data = segmentation_data.replace({'Yes': 1, 'No': 0})


# In[ ]:


from sklearn.decomposition import PCA

# Perform PCA
pca = PCA()
segmentation_pca = pca.fit_transform(segmentation_data)

# Determine the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Principal Components')
plt.show()


# In[ ]:


# Based on the PCA plot, you can decide the number of clusters, e.g., 4
kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(segmentation_data)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the original DataFrame
data['KMeans_Cluster'] = cluster_labels


# In[ ]:


# Assuming you decided on 4 clusters
gmm = GaussianMixture(n_components=4, random_state=1234)
gmm.fit(segmentation_data)

# Get cluster labels from the GMM model
gmm_cluster_labels = gmm.predict(segmentation_data)

# Add cluster labels to the original DataFrame
data['GMM_Cluster'] = gmm_cluster_labels


# In[ ]:



import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
import seaborn as sns


# In[ ]:


mcdonalds['Like.n'] = 6 - mcdonalds['Like']


# In[ ]:


# Creating the formula for regression
independent_vars = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy',
                    'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
formula = 'Like.n ~ ' + ' + '.join(independent_vars)


# In[ ]:


# Fitting a finite mixture of linear regression models
k = 2  # Number of clusters
nrep = 10  # Number of random starts
np.random.seed(1234)
kmeans = KMeans(n_clusters=k, n_init=nrep)
mcdonalds['Cluster'] = kmeans.fit_predict(mcdonalds[independent_vars])


# In[ ]:


# Summary of cluster sizes
cluster_sizes = mcdonalds['Cluster'].value_counts()


# In[ ]:


# Creating and fitting linear regression models for each cluster
models = []
for cluster in range(k):
    subset = mcdonalds[mcdonalds['Cluster'] == cluster]
    model = smf.ols(formula, data=subset).fit()
    models.append(model)


# In[ ]:


# Extracting coefficients for each cluster
coefficients = pd.DataFrame()
for cluster, model in enumerate(models):
    coef = model.params.reset_index()
    coef.columns = ['Variable', f'Cluster{cluster+1}']
    coefficients = pd.concat([coefficients, coef.iloc[1:]])  # Exclude Intercept


# In[ ]:


# Plotting the coefficients
sns.barplot(data=coefficients, x='Cluster1', y='Variable', hue='Cluster')
plt.title('Regression Coefficients for Each Cluster')
plt.xlabel('Coefficients')
plt.ylabel('Variables')
plt.show()


# In[ ]:


# Hierarchical clustering for segment profile plot
from scipy.cluster.hierarchy import linkage, dendrogram
segment_variables = ['Cluster1', 'Cluster2']
segment_data = mcdonalds[segment_variables]
segment_clusters = linkage(segment_data.T, method='ward')
dendrogram(segment_clusters, labels=segment_variables)
plt.title('Segment Profile Plot')
plt.xlabel('Segments')
plt.ylabel('Distance')
plt.show()

# Mosaic plot for cross-tabulation of segment membership and 'Like' variable
from statsmodels.graphics.mosaicplot import mosaic
mosaic(mcdonalds, ['Cluster', 'Like'])
plt.title('Segment vs. Like (Mosaic Plot)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




