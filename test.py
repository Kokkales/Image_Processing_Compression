import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute LBP and Contrast for an image
def compute_lbp_and_contrast(image):
    # Convert image to grayscale
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pad the image with zeros
    M, N = I.shape
    Inew = np.zeros((M+2, N+2), dtype=np.float64)
    Inew[1:M+1, 1:N+1] = I.astype(np.float64)

    # Initialize arrays
    C = np.zeros((M, N), dtype=np.float64)
    LBP = np.zeros_like(Inew, dtype=np.uint8)

    # Compute LBP and Contrast
    for i in range(1, M+1):
        for j in range(1, N+1):
            T = Inew[i, j]
            Maska = Inew[i-1:i+2, j-1:j+2].copy()
            Maska[1, 1] = 0
            C2 = np.sum(Maska)
            B = np.where(Maska > T)
            empty_B = len(B[0])

            if empty_B > 0:
                C1 = np.sum(Maska[B])
                Maska[:,:] = 0
                Maska[B] = 1
                number_1 = np.sum(Maska)
            else:
                C1 = 0
                Maska[:,:] = 0
                number_1 = 0

            LBP[i, j] = Maska[0, 0]*2**7 + Maska[0, 1]*2**6 + Maska[0, 2]*2**5 \
                        + Maska[1, 2]*2**4 + Maska[2, 2]*2**3 + Maska[2, 1]*2**2 \
                        + Maska[2, 0]*2**1 + Maska[1, 0]*2**0

            # Contrast calculation
            C2 = C2 - C1
            if number_1 > 0:
                if 8 - number_1 > 0:
                    C[i-1, j-1] = (C1 / number_1) - (C2 / (8 - number_1))
                else:
                    C[i-1, j-1] = C1 / number_1
            else:
                C[i-1, j-1] = -(C2 / (8 - number_1))

    # Prepare LBP image
    LBP = LBP[1:M+1, 1:N+1].astype(np.uint8)

    return LBP, C

# Function to extract normalized histogram of brightness
def extract_normalized_histogram(image, bins=256):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_norm = hist / np.sum(hist)  # Normalize histogram
    hist_norm = hist_norm.flatten()
    return hist_norm

# Directory containing categories, each containing images
dataset_path = './images/Villains'

# Iterate over each category in the dataset
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    # Ensure we're dealing with directories
    if os.path.isdir(category_path):
        print(f"Category: {category}")

        # Get the list of images in this category
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]

        # Select the first image in the category
        if images:
            image_filename = images[0]  # Select the first image
            image_path = os.path.join(category_path, image_filename)

            # Read the image
            image = cv2.imread(image_path)

            # Compute LBP and Contrast
            LBP, C = compute_lbp_and_contrast(image)

            # Compute histograms
            hist_brightness = extract_normalized_histogram(image)
            hist_lbp = np.histogram(LBP.ravel(), bins=256, range=[0, 256], density=True)[0]

            # Plotting
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            plt.title(f'Brightness Histogram - {image_filename}')
            plt.plot(hist_brightness)

            plt.subplot(2, 2, 2)
            plt.title(f'LBP Histogram - {image_filename}')
            plt.plot(hist_lbp)

            plt.subplot(2, 2, 3)
            plt.title(f'Original Image - {image_filename}')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')

            plt.subplot(2, 2, 4)
            plt.title(f'LBP Image - {image_filename}')
            plt.imshow(LBP, cmap='gray')

            # plt.tight_layout()
            plt.show()