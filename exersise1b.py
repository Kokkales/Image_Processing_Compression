import cv2
import numpy as np
import matplotlib.pyplot as plt

# Βήμα 1: Φορτώνουμε την εικόνα
image = cv2.imread('./images/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Τιμές για τα threshold και sigma
threshold_values = [1, 3, 6]
sigma_values = [1.0, 2.0, 3.0]

# Υπολογίζουμε τον αριθμό των υποπινάκων
num_plots = len(threshold_values) * len(sigma_values)

# Δημιουργούμε το σχήμα
plt.figure(figsize=(15, 10))

plot_counter = 1

for sigma in sigma_values:
    # Βήμα 2: Εφαρμόζουμε Gaussian Blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Βήμα 3: Υπολογίζουμε τις δεύτερες παραγώγους (Laplacian)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    for threshold in threshold_values:
        # Βήμα 4: Εφαρμόζουμε κατώφλι (threshold)
        edges = np.zeros_like(laplacian)
        edges[np.abs(laplacian) > threshold] = 255

        # Προσθέτουμε κάθε υποπίνακα στην κύρια εικόνα
        plt.subplot(len(sigma_values), len(threshold_values), plot_counter)
        plt.title(f'Th={threshold}, σ={sigma}', fontsize=10)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plot_counter += 1

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Edge Detection using LoG with Various Threshold and Sigma Values', fontsize=16)
plt.savefig('./resultsOne/edgeDetectionSecondDerivative.jpg')
plt.show()
