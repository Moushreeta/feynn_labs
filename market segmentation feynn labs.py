#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# distance


# In[1]:


import numpy as np
from scipy.spatial import distance

# Create a numpy array with vacation activity data (similar to the R data)
data = np.array([
    [100, 0, 0],
    [100, 0, 0],
    [60, 0, 0],
    [70, 40, 0],
    [80, 0, 0],
    [0, 0, 20],
    [50, 20, 30]
])

# Calculate Euclidean distance
euclidean_distances = distance.pdist(data, metric='euclidean')
euclidean_matrix = distance.squareform(euclidean_distances)

# Calculate Manhattan (Absolute) distance
manhattan_distances = distance.pdist(data, metric='cityblock')
manhattan_matrix = distance.squareform(manhattan_distances)

# Print the distance matrices (rounded for readability)
print(np.round(euclidean_matrix, 2))
print(np.round(manhattan_matrix, 2))


# In[4]:


'''
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data (assuming it's stored in a CSV file)
risk_data = pd.read_csv("tourist_risk_data.csv")

# Perform hierarchical clustering using Manhattan distance and complete linkage
risk_dist = sch.distance.pdist(risk_data, metric="cityblock")
risk_linkage = sch.linkage(risk_dist, method="complete")

# Generate the dendrogram
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(risk_linkage)
plt.title("Dendrogram")
plt.xlabel("Respondents")
plt.ylabel("Distance")
plt.show()

# Cut the dendrogram into six segments
num_segments = 6
risk_clusters = sch.fcluster(risk_linkage, num_segments, criterion="maxclust")

# Calculate the mean values for each category within each cluster
cluster_means = risk_data.groupby(risk_clusters).mean()

# Visualize the cluster characteristics using a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=cluster_means.index, y=cluster_means["Recreational"], palette="viridis")
plt.title("Mean Recreational Risk by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Mean Recreational Risk")
plt.show()
'''


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(1234)
# Generate artificial data with three clusters
n_samples = 500
centers = [(8, 8), (5, 5), (2, 2)]
data = np.concatenate([np.random.normal(center, 1, (n_samples // len(centers), 2)) for center in centers])


# In[ ]:


# Perform k-means clustering for different numbers of clusters (2 to 8)
k_values = range(2, 9)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(data)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Within-Cluster Distances')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Based on the elbow plot, choose the optimal number of clusters (e.g., 3)
optimal_k = 3

# Perform k-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=1234)
kmeans.fit(data)

# Get cluster assignments for each data point
cluster_assignments = kmeans.labels_

# Add cluster assignments to the original data (if needed)
data_with_clusters = pd.DataFrame(data, columns=['Features', 'Price'])
data_with_clusters['Cluster'] = cluster_assignments


# In[ ]:


library("kohonen")
set.seed(1234)
risk.som <- som(risk, somgrid(5, 5, "rect"))
plot(risk.som, palette.name = flxPalette, main = "")


# In[ ]:


import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Sample data (replace with your data)
data = np.random.rand(500, 5)

# Step 1: Partitioning (K-means)
k_partitioning = 30  # Choose a large number of clusters
kmeans = KMeans(n_clusters=k_partitioning, random_state=1234, n_init=10)
kmeans.fit(data)
cluster_centers = kmeans.cluster_centers_

# Step 2: Hierarchical Clustering
distances = pdist(cluster_centers, metric='euclidean')
linkage_matrix = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='complete').fit(distances)

# Combine the original data with the segmentation solution
cluster_memberships = kmeans.predict(data)

# Now, you can use 'cluster_memberships' to assign data points to segments


# In[ ]:


import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample data (replace with your data)
data = np.random.rand(2961, 27)

# Number of bootstrap samples and clusters
b = 50
k_partitioning = 10

# Step 1: Bootstrapping
bootstrap_samples = [np.random.choice(data.shape[0], size=data.shape[0], replace=True) for _ in range(b)]

# Initialize lists to store cluster centroids
cluster_centroids = []

# Step 2: Partitioning Clustering
for sample in bootstrap_samples:
    sample_data = data[sample]
    kmeans = KMeans(n_clusters=k_partitioning, random_state=1234, n_init=10)
    kmeans.fit(sample_data)
    cluster_centroids.extend(kmeans.cluster_centers_)

# Step 3: Derived Data Set
derived_data = np.array(cluster_centroids)

# Step 4: Hierarchical Clustering
distances = pdist(derived_data, metric='euclidean')
linkage_matrix = linkage(distances, method='average')  # Adjust linkage method if needed

# Step 5: Final Segmentation (Determine cut point on dendrogram)
dendrogram(linkage_matrix)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate or load your data, assuming 'X' is your data matrix

# Define the number of segments (components)
n_components = 3

# Create a GMM model
gmm = GaussianMixture(n_components=n_components, covariance_type='full')

# Fit the model to your data
gmm.fit(X)

# Get the predicted segment labels for each data point
segment_labels = gmm.predict(X)

# Visualize the segmentation results (you may need to adjust this based on your data)
plt.scatter(X[:, 0], X[:, 1], c=segment_labels, cmap='viridis')
plt.show()


# In[ ]:


# Australian Vacation Motives Example: Model-Based Segmentation


# In[ ]:


pip install scikit-learn


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture



# Define the number of segments (components)
n_components = 3  # Adjust this as needed

# Create a GMM model
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=123)

# Fit the model to your data
gmm.fit(vacmet)

# Get the predicted segment labels for each data point
segment_labels = gmm.predict(vacmet)

# Visualize the segmentation results (customize based on your dataset)

from pandas.plotting import scatter_matrix
import pandas as pd

# Assuming 'vacmet' is a pandas DataFrame
scatter_matrix(pd.DataFrame(vacmet, columns=["Obligation", "NEP", "Vacation.Behaviour"]), c=segment_labels, alpha=0.5, figsize=(10, 10), diagonal='hist')
plt.show()


# In[ ]:


# Austrian Winter Vacation Activities Example: Model-Based Segmentation for Binary Data


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Select the binary columns of interest
winterActiv_binary = winterActiv.iloc[:, 3:]  # Assuming columns 4 and onward are binary

# Define the number of segments (components)
n_components = 5  # Adjust this as needed

# Create a GMM model
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=123)

# Fit the model to your binary data
gmm.fit(winterActiv_binary)

# Get the predicted segment labels for each data point
segment_labels = gmm.predict(winterActiv_binary)

# Visualize the segmentation results (customize based on your dataset)
# We'll use a scatter plot matrix (pair plot) for visualization
scatter_matrix(winterActiv_binary, c=segment_labels, alpha=0.5, figsize=(10, 10), diagonal='hist')
plt.show()


# In[ ]:


# Finite Mixtures of Regressions Example in Python


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Simulate an artificial dataset similar to your theme park example
np.random.seed(1234)
n_samples = 500
rides = np.random.uniform(0, 50, n_samples)
epsilon = np.random.normal(0, 2, n_samples)
segment_1 = rides + epsilon
segment_2 = 0.0125 * rides**2 + epsilon

# Combine the two segments into one dataset
data = np.concatenate((segment_1, segment_2))

# Create an array to indicate the segment membership for each data point
segment_labels = np.array([0] * n_samples + [1] * n_samples)

# Create a scatter plot of the data points
plt.scatter(rides, data, c=segment_labels, cmap='viridis')
plt.xlabel('Number of Rides')
plt.ylabel('Willingness to Pay')
plt.title('Artificial Theme Park Data')
plt.show()

# Create a Gaussian Mixture Model (GMM) with 2 components (segments)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=123)

# Fit the GMM to the data
gmm.fit(np.column_stack((rides, rides**2)))

# Get the predicted segment labels for each data point
predicted_labels = gmm.predict(np.column_stack((rides, rides**2)))

# Visualize the segmentation results
plt.scatter(rides, data, c=predicted_labels, cmap='viridis')
plt.xlabel('Number of Rides')
plt.ylabel('Willingness to Pay')
plt.title('Segmentation using GMM')
plt.show()

# Extract the parameters of the fitted segments
means = gmm.means_
covariances = gmm.covariances_


# In[ ]:


# Finite Mixtures of Regressions Example in Python for Australian Travel Motives


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the Australian travel motives dataset
from flexclust import flexclust
data = flexclust::vacmotdesc
data = data.dropna(subset=["Obligation", "NEP", "Vacation.Behaviour"])

# Standardize the independent variables (moral obligation and NEP score)
scaler = StandardScaler()
data[["Obligation", "NEP"]] = scaler.fit_transform(data[["Obligation", "NEP"]])

# Fit a single linear regression model
envir_lm = LinearRegression()
X = data[["Obligation", "NEP"]]
y = data["Vacation.Behaviour"]
envir_lm.fit(X, y)

# Print the summary of the linear regression
print("Single Linear Regression Model:")
print("Intercept:", envir_lm.intercept_)
print("Coefficient for Obligation:", envir_lm.coef_[0])
print("Coefficient for NEP:", envir_lm.coef_[1])
print()

# Fit a Gaussian Mixture Model (GMM) to explore segments
gmm = GaussianMixture(n_components=2, random_state=123)
X = data[["Obligation", "NEP"]]
gmm.fit(X)

# Get the predicted segment labels for each data point
predicted_labels = gmm.predict(X)

# Visualize the segmentation results
plt.scatter(data["Obligation"], data["NEP"], c=predicted_labels, cmap='viridis')
plt.xlabel('Obligation')
plt.ylabel('NEP')
plt.title('Segmentation using GMM')
plt.show()

# Split the data into segments based on GMM labels
segment_1_data = data[predicted_labels == 0]
segment_2_data = data[predicted_labels == 1]

# Fit linear regression models for each segment
segment_1_lm = LinearRegression()
segment_2_lm = LinearRegression()

X_segment_1 = segment_1_data[["Obligation", "NEP"]]
y_segment_1 = segment_1_data["Vacation.Behaviour"]
segment_1_lm.fit(X_segment_1, y_segment_1)

X_segment_2 = segment_2_data[["Obligation", "NEP"]]
y_segment_2 = segment_2_data["Vacation.Behaviour"]
segment_2_lm.fit(X_segment_2, y_segment_2)

# Print the summaries of the linear regression models for each segment
print("Segment 1 Linear Regression Model:")
print("Intercept:", segment_1_lm.intercept_)
print("Coefficient for Obligation:", segment_1_lm.coef_[0])
print("Coefficient for NEP:", segment_1_lm.coef_[1])
print()

print("Segment 2 Linear Regression Model:")
print("Intercept:", segment_2_lm.intercept_)
print("Coefficient for Obligation:", segment_2_lm.coef_[0])
print("Coefficient for NEP:", segment_2_lm.coef_[1])


# In[ ]:


from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralBiclustering
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
data, rows, columns = make_biclusters(shape=(300, 300), n_clusters=4, noise=5, shuffle=False)

# Apply biclustering algorithm
model = SpectralBiclustering(n_clusters=4, method='log', random_state=0)
model.fit(data)

# Get row and column labels
row_labels = np.argsort(model.row_labels_)
column_labels = np.argsort(model.column_labels_)

# Visualize the results
plt.matshow(data[row_labels][:, column_labels], cmap=plt.cm.Blues)
plt.title("Biclusters")
plt.show()


# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Define the VSBD function
def vsbd(data, centers, phi=1.0, V=4, delta=0.5):
    # Step 1: Select a subset of observations
    num_obs = data.shape[0]
    subset_size = int(phi * num_obs)
    subset_indices = np.random.choice(num_obs, size=subset_size, replace=False)
    subset_data = data[subset_indices, :]

    # Step 2: Find the best subset of V variables
    best_variables = None
    best_inertia = float('inf')
    for _ in range(500):  # Number of random initializations in step 2
        random_indices = np.random.choice(data.shape[1], size=V, replace=False)
        kmeans = KMeans(n_clusters=centers, n_init=10, random_state=0).fit(subset_data[:, random_indices])
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_variables = random_indices

    # Step 3 and 4: Add variables one by one
    remaining_variables = list(set(range(data.shape[1])) - set(best_variables))
    while remaining_variables:
        best_variable = None
        best_inertia_increase = float('inf')
        for var in remaining_variables:
            new_variable_set = np.append(best_variables, var)
            kmeans = KMeans(n_clusters=centers, n_init=10, random_state=0).fit(subset_data[:, new_variable_set])
            new_inertia = kmeans.inertia_
            inertia_increase = new_inertia - best_inertia
            if inertia_increase < best_inertia_increase:
                best_inertia_increase = inertia_increase
                best_variable = var
        if best_inertia_increase < delta * (subset_size / 4):
            best_variables = np.append(best_variables, best_variable)
            remaining_variables.remove(best_variable)
        else:
            break

    # Step 5: Cluster with selected variables
    kmeans_final = KMeans(n_clusters=centers, n_init=10, random_state=0).fit(data[:, best_variables])

    # Step 6: Interpretation
    return kmeans_final, best_variables

# Example usage
from sklearn.datasets import make_blobs

# Generate synthetic binary data
data, _ = make_blobs(n_samples=300, centers=6, random_state=0, cluster_std=0.60)
data[data < 0] = 0  # Make data binary

# Apply VSBD algorithm
centers = 6  # Number of clusters
kmeans_final, selected_variables = vsbd(data, centers)

# Get cluster assignments
cluster_labels = kmeans_final.labels_

# Print selected variables and cluster assignments
print("Selected Variables:", selected_variables)
print("Cluster Assignments:", cluster_labels)


# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Load your data (replace this with your data loading code)
# risk_data = ...

# Perform bootstrapping for k-means clustering
num_bootstrap_samples = 100
num_segments_range = range(2, 10)
rand_indices = np.zeros((num_bootstrap_samples, len(num_segments_range)))

for i in range(num_bootstrap_samples):
    bootstrap_indices = np.random.choice(len(risk_data), len(risk_data), replace=True)
    for j, num_segments in enumerate(num_segments_range):
        kmeans = KMeans(n_clusters=num_segments, random_state=1234)
        cluster_labels = kmeans.fit_predict(risk_data[bootstrap_indices])
        rand_indices[i, j] = adjusted_rand_score(true_labels, cluster_labels)  # Replace true_labels

# Create a global stability boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(rand_indices, labels=num_segments_range)
plt.xlabel("Number of Segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Global Stability Boxplot")
plt.ylim(0, 1)
plt.grid(True)
plt.show()


# In[ ]:




