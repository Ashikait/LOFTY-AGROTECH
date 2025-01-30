import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_leaf_and_remove_background(image_path):
    # Load the original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert image to HSV and create a mask for green regions (leaf color range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply mask to keep only the leaf
    leaf_only = cv2.bitwise_and(img, img, mask=mask)
    
    return img_rgb, leaf_only, mask

def edge_detection_and_closed_curve(img, mask):
    # Apply Canny edge detection
    edges = cv2.Canny(mask, threshold1=50, threshold2=150)

    # Find contours on the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask to draw closed curves (contours)
    closed_curve_mask = np.zeros_like(mask)
    cv2.drawContours(closed_curve_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Apply the mask to crop the leaf region from the original image
    cropped_leaf = cv2.bitwise_and(img, img, mask=closed_curve_mask)
    
    return edges, cropped_leaf
def edge_detection(image_path):
    # Load the image in color (BGR format)
    img_bgr = cv2.imread(image_path)
    
    # Convert the image to grayscale for edge detection
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    
    # Convert the edges to 3-channel color format (BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Color the detected edges (highlight in red for visibility)
    img_with_edges = img_bgr.copy()
    img_with_edges[edges == 255] = [0, 0, 255]  # Red color for the edges
    
    return img_bgr, edges_colored, img_with_edges

# Path to the leaf image
image_path = r"C:\Users\acer\Downloads\PADDY IMAGES\bacterial_leaf_blight\100234.jpg"  # Replace with your image path
# Step 1: Perform edge detection in color
original_img, edges_colored, img_with_edges = edge_detection(image_path)

# Display the results
plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display in matplotlib
plt.title("Original Image")
plt.axis('off')

# Display the edge detection with the original color
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_with_edges, cv2.COLOR_BGR2RGB))  # Show edges highlighted in red on the original image
plt.title("Edge Detection in Color (Red Edges)")
plt.axis('off')
def superimpose_images(original_img, edges_img):
    # Convert edge image to 3-channel format
    edges_img_color = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)
    
    # Superimpose the edge-detected image on the original
    superimposed_img = cv2.addWeighted(original_img, 0.7, edges_img_color, 0.3, 0)
    
    return superimposed_img

def calculate_disease_severity(mask):
    # Calculate the percentage of affected area based on the mask
    total_area = mask.shape[0] * mask.shape[1]
    affected_area = np.sum(mask == 255)  # Count white pixels (affected areas)
    severity_ratio = affected_area / total_area
    
    return severity_ratio * 100  # Return as a percentage

def classify_disease(severity_ratio, threshold=15.0):
    # Classify as healthy or diseased based on severity threshold
    if severity_ratio < threshold:
        return "Healthy"
    else:
        return "Diseased"

# Path to the leaf image
image_path = r"C:\Users\acer\Downloads\PADDY IMAGES\bacterial_leaf_blight\100234.jpg"  # Replace with your image path

# Step 1: Detect leaf and remove background
original_img, leaf_only, mask = detect_leaf_and_remove_background(image_path)

# Step 2: Perform edge detection and ensure closed curves
edges, cropped_leaf = edge_detection_and_closed_curve(original_img, mask)

# Step 3: Superimpose edge-detected image on the original
superimposed_img = superimpose_images(original_img, edges)

# Step 4: Calculate disease severity
severity_ratio = calculate_disease_severity(mask)

# Step 5: Classify the leaf based on severity
classification = classify_disease(severity_ratio)

# Display the results
#plt.figure(figsize=(15, 5))
#plt.subplot(1, 3, 1)
#plt.imshow(original_img)
#plt.title("Original Image")
#plt.axis('off')

#plt.subplot(1, 3, 2)
#plt.imshow(edges, cmap='gray')
#plt.title("Edge Detection (Closed Curves)")
#plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title(f"Superimposed Image\nClassification: {classification} ({severity_ratio:.2f}% Severity)")
plt.axis('off')

plt.show()

print(f"Leaf Classification: {classification}")
print(f"Disease Severity: {severity_ratio:.2f}%")
