import os
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np

def perform_kmeans_clustering(image_folder, num_clusters):
    # Load images and convert them to feature vectors
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((256, 256))  # Resize the image to a consistent size
        img = np.array(img)  # Convert the image to a numpy array
        images.append(img.flatten())  # Flatten the array and add it to the list of images
    X = np.array(images)  # Convert the list of images to a numpy array

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform dimensionality reduction
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_pca)

    # Create directories for each cluster
    cluster_folder = os.path.join(image_folder, "clusters")
    os.makedirs(cluster_folder, exist_ok=True)
    for i in range(num_clusters):
        cluster_dir = os.path.join(cluster_folder, f"cluster_{i+1}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Move images to their respective cluster folders
    for i, path in enumerate(image_paths):
        img_cluster = kmeans.labels_[i]
        cluster_dir = os.path.join(cluster_folder, f"cluster_{img_cluster+1}")
        shutil.copy(path, cluster_dir)

    print("Clustering completed successfully.")
    
    
image_folder = "../results/images"
num_clusters = 13

perform_kmeans_clustering(image_folder, num_clusters)