import cv2
import matplotlib.pyplot as plt

# Load the image
imagePath = './images/lighthouse.bmp'
image = cv2.imread(imagePath)

# Convert the image to grayscale
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(grayImage, 100, 200)  # Default thresholds in OpenCV, in Octave is 0.1 and 0.2

# Plot the original image and the edge-detected image
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
plt.savefig('./resultsOneC/edgeDetectionCanny.jpg')
plt.show()
