import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

# LBP orismata
RADIUS = 1
N_POINTS = 8 * RADIUS

villainsPath = './images/Villains'  # Προσαρμόστε το path αναλόγως

# Normalization histogram
def normalizeHistogram(hist):
    return hist / np.sum(hist)

# extract histogam
def extractNormalizedHistogram(imageGray):
    hist = cv2.calcHist([imageGray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = normalizeHistogram(hist)
    return hist

# extract LBP histogram
def extractNormalizedLbpHistogram(imageGray):
    lbp = local_binary_pattern(imageGray, N_POINTS, RADIUS, method='default')
    (hist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = normalizeHistogram(hist)
    return hist

# L1
def computeLOneDistance(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))

# L2
def computeLTwoDistance(hist1, hist2):
    return np.sqrt(np.sum((hist1 - hist2) ** 2))

# get all images
def getAllImagePaths(base_path):
    imagePaths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                imagePaths.append(os.path.join(root, file))
    return imagePaths
imagePaths = getAllImagePaths(villainsPath)

# Λίστες για αποθήκευση των χαρακτηριστικών
brightnessHistograms = []
lbpHistograms = []
imageFiles = []

# process each image
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # featire extraction
    brightness_hist = extractNormalizedHistogram(imageGray)
    lbpHist = extractNormalizedLbpHistogram(imageGray)

    brightnessHistograms.append(brightness_hist)
    lbpHistograms.append(lbpHist)
    imageFiles.append(imagePath)

# similarity calculation για όλους τους συνδυασμούς
num_images = len(imageFiles)
results = []

for i in range(num_images):
    for j in range(i + 1, num_images):
        lOneBrightness = computeLOneDistance(brightnessHistograms[i], brightnessHistograms[j])
        lTwoBrightness = computeLTwoDistance(brightnessHistograms[i], brightnessHistograms[j])
        lOneLbp = computeLOneDistance(lbpHistograms[i], lbpHistograms[j])
        lTwoLbp = computeLTwoDistance(lbpHistograms[i], lbpHistograms[j])

        results.append({
            'image1': imageFiles[i],
            'image2': imageFiles[j],
            'lOneBrightness': lOneBrightness,
            'lTwoBrightness': lTwoBrightness,
            'lOneLbp': lOneLbp,
            'lTwoLbp': lTwoLbp
        })

for result in results:
    print(f"Image 1: {result['image1']}")
    print(f"Image 2: {result['image2']}")
    print(f"L1 distance (brightness): {result['lOneBrightness']}")
    print(f"L2 distance (brightness): {result['lTwoBrightness']}")
    print(f"L1 distance (LBP): {result['lOneLbp']}")
    print(f"L2 distance (LBP): {result['lTwoLbp']}")
    print('-' * 50)

# Γ. SIMILAR IMAGES
def getTopKSimilarImages(queryHist, histograms, imageFiles, k, distanceMetric):
    distances = []
    for i, hist in enumerate(histograms):
        if not np.array_equal(queryHist, hist):  # Εξαίρεση της ίδιας της εικόνας
            distance = distanceMetric(queryHist, hist)
            distances.append((distance, imageFiles[i]))
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def select_random_images(imageFiles, num_images):
    selectedImages = []
    categories = set()

    while len(selectedImages) < num_images:
        imageFile = random.choice(imageFiles)
        category = os.path.basename(os.path.dirname(imageFile))
        if category not in categories:
            selectedImages.append(imageFile)
            categories.add(category)

    return selectedImages

queryImages = select_random_images(imageFiles, 5)

avgDistances = {
    'brightness_L1': [],
    'brightness_L2': [],
    'lbp_L1': [],
    'lbp_L2': []
}

def extract_category_name(file_path):
    # Use regular expression to match the category name
    # match = re.search(r'images[\\/](.*?)[\\/]', file_path)
    match = re.search(r'Villains[\\/](.*?)[\\/]', file_path)
    if match:
        return match.group(1)
    else:
        return None
# Υπολογισμός των top-5 πιο όμοιων εικόνων για κάθε query image
k = 5

def plot_similar_images(queryImagePath, similar_images, title):
    """
    Plots the query image alongside the top 5 similar images.

    Parameters:
    - queryImagePath: Path to the query image.
    - similar_images: List of tuples (distance, imagePath) of similar images.
    - title: Title for the plot.
    """
    fig, axes = plt.subplots(1, 6, figsize=(10, 3))
    fig.suptitle(title, fontsize=16)

    # Plot the query image
    query_img = mpimg.imread(queryImagePath)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # Plot the similar images
    for i, (distance, imagePath) in enumerate(similar_images):
        similar_img = mpimg.imread(imagePath)
        axes[i + 1].imshow(similar_img)
        axes[i + 1].set_title(f"Dist: {distance:.2f}")
        axes[i + 1].axis('off')
    plt.savefig(f"./resultsFour/{extract_category_name(queryImagePath)}_{title}.jpg")
    plt.show()

def plotHistograms(lbp_histograms, brightness_histograms, queryImagePath):
    # Παράδειγμα εμφάνισης αποτελεσμάτων για την πρώτη εικόνα
    first_brightness_hist = brightness_histograms
    first_lbp_hist = lbp_histograms

    # Σχεδίαση των ιστογραμμάτων σε ένα διάγραμμα
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Ιστόγραμμα φωτεινότητας
    ax[0].bar(range(256), first_brightness_hist, color='gray')
    ax[0].set_title('Normalized Brightness Histogram')
    ax[0].set_xlim([0, 255])

    # Ιστόγραμμα LBP
    ax[1].bar(range(256), first_lbp_hist, color='gray')
    ax[1].set_title('Normalized LBP Histogram')
    ax[1].set_xlim([0, 255])

    # Εμφάνιση του διαγράμματος
    plt.tight_layout()
    plt.savefig(f'./resultsFour/{extract_category_name(queryImagePath)}_histograms.jpg')
    plt.show()


for queryImagePath in queryImages:
    queryImage = cv2.imread(queryImagePath)
    queryImageGray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features
    queryBrightnessHist = extractNormalizedHistogram(queryImageGray)
    queryLbpHist = extractNormalizedLbpHistogram(queryImageGray)
    plotHistograms(queryLbpHist, queryBrightnessHist, queryImagePath)

    # L1 brightness
    top_5_brightness_l1 = getTopKSimilarImages(queryBrightnessHist, brightnessHistograms, imageFiles, k, computeLOneDistance)
    avg_distance_brightness_l1 = np.mean([d[0] for d in top_5_brightness_l1])
    avgDistances['brightness_L1'].append(avg_distance_brightness_l1)

    # L2 brightness
    top_5_brightness_l2 = getTopKSimilarImages(queryBrightnessHist, brightnessHistograms, imageFiles, k, computeLTwoDistance)
    avg_distance_brightness_l2 = np.mean([d[0] for d in top_5_brightness_l2])
    avgDistances['brightness_L2'].append(avg_distance_brightness_l2)

    # L1 LPB
    top_5_lbp_l1 = getTopKSimilarImages(queryLbpHist, lbpHistograms, imageFiles, k, computeLOneDistance)
    avg_distance_lbp_l1 = np.mean([d[0] for d in top_5_lbp_l1])
    avgDistances['lbp_L1'].append(avg_distance_lbp_l1)

    # L2 LBP
    top_5_lbp_l2 = getTopKSimilarImages(queryLbpHist, lbpHistograms, imageFiles, k, computeLTwoDistance)
    avg_distance_lbp_l2 = np.mean([d[0] for d in top_5_lbp_l2])
    avgDistances['lbp_L2'].append(avg_distance_lbp_l2)

    print(f"Query Image: {queryImagePath}")
    print("Top 5 similar images based on brightness histogram with L1 distance:")
    for distance, imagePath in top_5_brightness_l1:
        print(f"Image: {imagePath}, Distance: {distance}")

    print("Top 5 similar images based on brightness histogram with L2 distance:")
    for distance, imagePath in top_5_brightness_l2:
        print(f"Image: {imagePath}, Distance: {distance}")

    print("Top 5 similar images based on LBP histogram with L1 distance:")
    for distance, imagePath in top_5_lbp_l1:
        print(f"Image: {imagePath}, Distance: {distance}")

    print("Top 5 similar images based on LBP histogram with L2 distance:")
    for distance, imagePath in top_5_lbp_l2:
        print(f"Image: {imagePath}, Distance: {distance}")

    print('======================================================================')

    plot_similar_images(queryImagePath, top_5_brightness_l1, "Top 5 similar images based on brightness histogram with L1 distance")
    plot_similar_images(queryImagePath, top_5_brightness_l2, "Top 5 similar images based on brightness histogram with L2 distance")
    plot_similar_images(queryImagePath, top_5_lbp_l1, "Top 5 similar images based on LBP histogram with L1 distance")
    plot_similar_images(queryImagePath, top_5_lbp_l2, "Top 5 similar images based on LBP histogram with L2 distance")


overallAvgDistances = {key: np.mean(value) for key, value in avgDistances.items()}
print("Overall average distances for each combination:")
for key, value in overallAvgDistances.items():
    print(f"{key}: {value}")
best_combination = min(overallAvgDistances, key=overallAvgDistances.get)
print(f"The best combination of feature and metric is: {best_combination}")