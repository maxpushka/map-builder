from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hashlib
import os


def pca_dbscan(feature_list):
    # Scale and reduce dimensions
    X = np.array(feature_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    # 3D PCA
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)

    # Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_pca_3d)
    dbscan_labels = dbscan.labels_
    return dbscan_labels, X_pca_2d, X_pca_3d


# Function to calculate hash for a single image file
def _calculate_single_image_hash(image_path):
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


# Function to calculate combined hash for all images concurrently
def _calculate_image_hash_concurrently(image_paths):
    hash_md5 = hashlib.md5()
    sorted_paths = sorted(image_paths) # Sort image paths to ensure consistent order
    partial_hashes = {}

    # Calculate each image's hash concurrently
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_calculate_single_image_hash, path): path
            for path in sorted_paths
        }

        # Store results in a dictionary to preserve path-order mapping
        for future in as_completed(futures):
            path = futures[future]
            partial_hashes[path] = future.result()

    # Combine partial hashes in the sorted order of paths
    for path in sorted_paths:
        hash_md5.update(partial_hashes[path].encode())  # Update with each partial hash

    return hash_md5.hexdigest()


image_dir = "alt"  # Directory containing all images
image_paths = [  # Get list of all image files in the directory
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
]

image_hash = _calculate_image_hash_concurrently(image_paths)
feature_file = f"features_{image_hash}.csv"  # CSV file to store/load features
