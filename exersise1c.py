import cv2
import matplotlib.pyplot as plt

imagePath = './images/bridge.bmp'
image = cv2.imread(imagePath)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ψαννυ
edges = cv2.Canny(grayImage, 100, 200)  # τα Default thresholds στην Octave είναι 0.1 και 0.2, όπου στην OpenCV της python αντιστοιχούν σε 100 και 200 αντίστοιχα

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
plt.savefig('./resultsOneC/edgeDetectionCannyGirlFaceBridge.jpg')
plt.show()
