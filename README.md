#Edge Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
image_path = r"C:\Users\acer\Downloads\leaf_image.jpeg"  
if os.path.exists(image_path):
    print("File exists:", image_path)
else:
    print("File not found at:", image_path)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blurred_image = cv2.GaussianBlur(image, (25, 25), sigmaX=30, sigmaY=30)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
dilated_edges = cv2.dilate(edges, kernel=np.ones((3, 3), np.uint8), iterations=1)
mask = cv2.bitwise_not(dilated_edges)
mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=10)
mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask_normalized = mask_3channel / 255.0
blended_image = (image * mask_normalized + blurred_image * (1 - mask_normalized)).astype(np.uint8)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
plt.figure(figsize=(15, 10))
plt.subplot(1, 4, 1)
#original Image
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")
#Edges Detected
plt.subplot(1, 4, 2)
plt.title("Edges Detected")
plt.imshow(edges, cmap="gray")
plt.axis("off")
# Final Image with Blurred Background
plt.subplot(1, 4, 3)
plt.title("Final Image (Blurred Background)")
plt.imshow(blended_image)
plt.axis("off")
# Sharpened Image
plt.subplot(1, 4, 4)
plt.title("Sharpened Edges")
plt.imshow(sharpened_image)
plt.axis("off")
plt.tight_layout()
plt.show()
