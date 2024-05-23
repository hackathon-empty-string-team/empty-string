# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# %%
# Load the results dictionary
with open('../data/result_feature_extraction/results_dict.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# Collect all conv_output entries into a single array and flatten each output
conv_outputs = []
for key, entries in results_dict['spectrograms'].items():
    for entry in entries:
        conv_outputs.append(entry['conv_output'].flatten())

conv_outputs = np.array(conv_outputs)

# Apply K-means clustering
# I am aware this is not what we are looking for. I just wanted to see what happens when applying k-means. 
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(conv_outputs)

# Add cluster labels to the results dictionary
for key, entries in results_dict['spectrograms'].items():
    for entry in entries:
        entry['cluster'] = kmeans.predict(entry['conv_output'].flatten().reshape(1, -1))[0]

# Visualize the clustering results using the first two principal components
pca = PCA(n_components=2)
conv_outputs_pca = pca.fit_transform(conv_outputs)

plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    plt.scatter(conv_outputs_pca[cluster_indices, 0], conv_outputs_pca[cluster_indices, 1], label=f'Cluster {i}')
plt.legend()
plt.title('K-means Clustering of Conv2D Outputs')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# %%

# %%

# %%
