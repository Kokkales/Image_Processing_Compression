import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/girlface.bmp', cv2.IMREAD_GRAYSCALE)

# th and sigma
thresholdValues = [1, 3, 6]
sigmaValues = [1.0, 2.0, 3.0]

plotCounter = 1
plt.figure(figsize=(15, 10))
for sigma in sigmaValues:
    # Gaussian Blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # 2nd derivetives
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    for threshold in thresholdValues:
        # set threshold
        edges = np.zeros_like(laplacian)
        edges[np.abs(laplacian) > threshold] = 255

        plt.subplot(len(sigmaValues), len(thresholdValues), plotCounter)
        plt.title(f'Th={threshold}, σ={sigma}', fontsize=10)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plotCounter += 1

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('./resultsOneB/edgeDetectionSecondDerivativeGirlFace.jpg')
plt.show()
