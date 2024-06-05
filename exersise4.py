import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import random

# Ορισμός των παραμέτρων για το LBP
radius = 1
n_points = 8 * radius

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_normalized_histogram(image, bins=256):
    histogram, bin_edges = np.histogram(image, bins=bins, range=(0, 256))
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + 1e-7)  # κανονικοποίηση
    return histogram

def calculate_lbp_histogram(image, radius, n_points, bins=256):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    histogram, _ = np.histogram(lbp, bins=np.arange(0, bins + 1), range=(0, bins))
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + 1e-7)  # κανονικοποίηση
    return histogram

def extract_features(image):
    gray_image = convert_to_grayscale(image)
    brightness_histogram = calculate_normalized_histogram(gray_image)
    lbp_histogram = calculate_lbp_histogram(gray_image, radius, n_points)
    return np.concatenate((brightness_histogram, lbp_histogram))

def metric_b1(f1, f2):
    return np.sum(np.abs(f1 - f2))

def metric_b2(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))

# Step 1: Select 5 random images from different categories
folder_path = './images/Villains/'
category_folders = os.listdir(folder_path)
random_images = []
for category_folder in category_folders:
    images_in_category_folder = os.listdir(os.path.join(folder_path, category_folder))
    random_image = random.choice(images_in_category_folder)
    random_images.append(cv2.imread(os.path.join(folder_path, category_folder, random_image)))

# Step 2-4: For each query image, compute similarity with all other images and rank them
for query_image in random_images:
    similarity_scores = []
    query_features = extract_features(query_image)  # Extract features for the query image

    for category_folder in category_folders:
        images_in_category_folder = os.listdir(os.path.join(folder_path, category_folder))
        for image_name in images_in_category_folder:
            image = cv2.imread(os.path.join(folder_path, category_folder, image_name))
            if not np.array_equal(image, query_image):
                image_features = extract_features(image)  # Extract features for the current image
                # Calculate similarity using one of the metrics (B1 or B2)
                similarity = metric_b1(query_features, image_features)
                similarity_scores.append((image_name, similarity))

    # Rank images based on similarity scores
    similarity_scores.sort(key=lambda x: x[1])

    # Step 5: Print the top 5 retrieval results
    print("Query Image:", query_image)
    print("Top 5 Retrieval Results:")
    for i, (retrieved_image_name, similarity) in enumerate(similarity_scores[:5], 1):
        print(f"{i}. Image: {retrieved_image_name}, Similarity: {similarity}")
