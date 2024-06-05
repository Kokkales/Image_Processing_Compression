import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Edge detection functions
def sobelEdgeDetection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    return sobel

def robertsCrossEdgeDetection(image):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    robertsx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(image, cv2.CV_64F, kernely)
    roberts = np.hypot(robertsx, robertsy)
    return roberts

def prewittEdgeDetection(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewittx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(image, cv2.CV_64F, kernely)
    prewitt = np.hypot(prewittx, prewitty)
    return prewitt

def kirschEdgeDetection(image):
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]
    kirsch = np.zeros_like(image, dtype=np.float64)
    for kernel in kernels:
        kirsch = np.maximum(kirsch, cv2.filter2D(image, cv2.CV_64F, kernel))
    return kirsch

def otsuThresholding(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Directory containing images
imageDir = './images/'
outputDir = './resultsOneA/'

# List of image filenames
imageFiles = ['lighthouse.bmp','bridge.bmp', 'girlFace.bmp']

# Iterate over each image
for filename in imageFiles:
    # Load the image
    imagePath = os.path.join(imageDir, filename)
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # Apply edge detection methods
    sobelEdges = sobelEdgeDetection(image)
    robertsEdges = robertsCrossEdgeDetection(image)
    prewittEdges = prewittEdgeDetection(image)
    kirschEdges = kirschEdgeDetection(image)

    # Apply Otsu's thresholding
    sobelBinary = otsuThresholding(sobelEdges.astype(np.uint8))
    robertsBinary = otsuThresholding(robertsEdges.astype(np.uint8))
    prewittBinary = otsuThresholding(prewittEdges.astype(np.uint8))

    # Display and save results for Sobel, Roberts, and Prewitt
    plt.figure(figsize=(18, 6))
    plt.subplot(141), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(142), plt.imshow(sobelBinary, cmap='gray'), plt.title('Sobel')
    plt.subplot(143), plt.imshow(robertsBinary, cmap='gray'), plt.title('Roberts')
    plt.subplot(144), plt.imshow(prewittBinary, cmap='gray'), plt.title('Prewitt')

    # Create output directory if it doesn't exist
    os.makedirs(outputDir, exist_ok=True)
    outputFilename = os.path.splitext(filename)[0] + '_edges.jpg'
    outputPath = os.path.join(outputDir, outputFilename)
    plt.savefig(outputPath)
    plt.show()

    # Apply Kirsch thresholding experiment
    thresholds = [50, 100, 150, 200, 250, 300]
    plt.figure(figsize=(18, 12))
    plt.subplot(3, 3, 1), plt.imshow(kirschEdges, cmap='gray'), plt.title('Kirsch Original')
    for i, t in enumerate(thresholds):
        _, kirschBinary = cv2.threshold(kirschEdges.astype(np.uint8), t, 255, cv2.THRESH_BINARY)
        plt.subplot(3, 3, i + 2), plt.imshow(kirschBinary, cmap='gray'), plt.title(f'Threshold {t}')
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(outputDir, exist_ok=True)
    outputFilename = os.path.splitext(filename)[0] + '_kirsch_thresholds.jpg'
    outputPath = os.path.join(outputDir, outputFilename)
    plt.savefig(outputPath)
    plt.show()
