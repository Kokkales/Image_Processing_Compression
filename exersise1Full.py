import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sobel Operator
def sobelEdgeDetection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    return sobel

# Roberts Cross Operator
def robertsCrossEdgeDetection(image):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    robertsx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(image, cv2.CV_64F, kernely)
    roberts = np.hypot(robertsx, robertsy)
    return roberts

# Prewitt Operator
def prewittEdgeDetection(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewittx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(image, cv2.CV_64F, kernely)
    prewitt = np.hypot(prewittx, prewitty)
    return prewitt

# Kirsch Operator
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

# Thresholding
def otsuThresholding(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Load the image
image = cv2.imread('./images/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Apply edge detection methods
sobelEdges = sobelEdgeDetection(image)
robertsEdges = robertsCrossEdgeDetection(image)
prewittEdges = prewittEdgeDetection(image)
kirschEdges = kirschEdgeDetection(image)

# Apply Otsu's thresholding
sobelBinary = otsuThresholding(sobelEdges.astype(np.uint8))
robertsBinary = otsuThresholding(robertsEdges.astype(np.uint8))
prewittBinary = otsuThresholding(prewittEdges.astype(np.uint8))

# Display results
plt.figure(figsize=(10, 8))

plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(232), plt.imshow(sobelBinary, cmap='gray'), plt.title('Sobel')
plt.subplot(233), plt.imshow(robertsBinary, cmap='gray'), plt.title('Roberts')
plt.subplot(234), plt.imshow(prewittBinary, cmap='gray'), plt.title('Prewitt')

# Kirsch thresholding experiment
thresholds = [50, 100, 150, 200, 250, 300]
plt.figure(figsize=(12, 8))
for i, t in enumerate(thresholds):
    _, kirschBinary = cv2.threshold(kirschEdges.astype(np.uint8), t, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 3, i+1)
    plt.imshow(kirschBinary, cmap='gray')
    plt.title(f'Kirsch Threshold {t}')

# Edge detection using LoG
threshold_values = [1, 3, 6]
sigma_values = [1.0, 2.0, 3.0]
num_plots = len(threshold_values) * len(sigma_values)

plt.figure(figsize=(15, 10))
plot_counter = 1

for sigma in sigma_values:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    for threshold in threshold_values:
        edges = np.zeros_like(laplacian)
        edges[np.abs(laplacian) > threshold] = 255
        plt.subplot(len(sigma_values), len(threshold_values), plot_counter)
        plt.title(f'Th={threshold}, σ={sigma}', fontsize=10)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plot_counter += 1

# Canny edge detection
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grayImage, 100, 200)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.tight_layout()

plt.show()
