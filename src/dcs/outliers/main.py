import cv2
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Directory containing all images
image_dir = "alt"  # Replace 'alt' with the full path to your directory if needed

# Get list of all image files in the directory
image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
]


# Function to extract features
def extract_features(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Resize image to a fixed size (e.g., 128x128)
    resized_image = cv2.resize(image, (128, 128))

    # Calculate mean and standard deviation of pixel intensities
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)

    # Edge detection using Canny
    edges = cv2.Canny(resized_image, 100, 200)
    edge_density = np.sum(edges) / (128 * 128)  # Normalized edge density

    # Histogram of pixel intensities (normalized)
    hist = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize histogram

    # Flatten histogram for feature vector
    hist_flattened = hist.flatten()

    # Combine all features
    features = np.array(
        [mean_intensity, std_intensity, edge_density] + hist_flattened.tolist()
    )
    return features

# Load feature list from file (optional)
feature_file = "features.csv"
if os.path.exists(feature_file):
    feature_list = np.loadtxt(feature_file, delimiter=",")

# Extract features for all images
if feature_list is None:
    feature_list = []
    for i, path in enumerate(image_paths):
        # measure progress
        start_time = time.time()
        features = extract_features(path)
        if features is not None:
            feature_list.append(features)
        if i % 100 == 0:
            iter_time = time.time() - start_time
            print(f"Processed {i} images for {iter_time:.2f}s, {len(image_paths) - i} more left")
    # Write feature list to a file (optional)
    np.savetxt(feature_file, feature_list, delimiter=",")

# Convert to numpy array
X = np.array(feature_list)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Apply PCA for dimensionality reduction
pca = PCA(n_components=20)  # Keep 20 components, adjust based on need
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering
num_clusters = 2  # Adjust based on dataset, can try more clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_pca)

# Get cluster labels
cluster_labels = kmeans.labels_

# Optional: Evaluate clustering with silhouette score
silhouette_avg = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize clusters in PCA-reduced space (first two components)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis")
plt.colorbar(label="Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Satellite Images")
plt.show()

# Identify potential outliers (small clusters)
# Find cluster with the smallest number of images, as it may indicate an outlier group
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
print("Cluster Sizes:", cluster_sizes)

# Mark images in the smallest cluster as potential outliers
outlier_cluster = min(cluster_sizes, key=cluster_sizes.get)
outliers = [
    image_paths[i]
    for i in range(len(cluster_labels))
    if cluster_labels[i] == outlier_cluster
]

print(f"Potential outlier images (cluster {outlier_cluster}):")
for outlier in outliers:
    print(outlier)
