import cv2
import numpy as np
import matplotlib.pyplot as plt

# Βήμα 1: Φορτώνουμε την εικόνα
image = cv2.imread('./images/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Τιμές για τα threshold και sigma
thresholdValues = [1, 3, 6]
sigmaValues = [1.0, 2.0, 3.0]

# Δημιουργούμε το σχήμα
plt.figure(figsize=(15, 10))

plotCounter = 1

for sigma in sigmaValues:
    # Βήμα 2: Εφαρμόζουμε Gaussian Blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Βήμα 3: Υπολογίζουμε τις δεύτερες παραγώγους (Laplacian)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    for threshold in thresholdValues:
        # Βήμα 4: Εφαρμόζουμε κατώφλι (threshold)
        edges = np.zeros_like(laplacian)
        edges[np.abs(laplacian) > threshold] = 255

        # Προσθέτουμε κάθε υποπίνακα στην κύρια εικόνα
        plt.subplot(len(sigmaValues), len(thresholdValues), plotCounter)
        plt.title(f'Th={threshold}, σ={sigma}', fontsize=10)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plotCounter += 1

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Edge Detection using LoG with Various Threshold and Sigma Values', fontsize=16)
plt.savefig('./resultsOneB/edgeDetectionSecondDerivative.jpg')
plt.show()
