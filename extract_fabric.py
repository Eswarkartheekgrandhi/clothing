import cv2
import numpy as np

# Load the extracted shirt image
shirt_img = cv2.imread("71CDIw+eTNL._AC_UY1100_.jpg")

# Get image dimensions
h, w, _ = shirt_img.shape

# Define patch size (increase for better quality)
patch_size = 300  

# Get center coordinates
center_x, center_y = w // 2, h // 2

# Crop a larger fabric patch
fabric_patch = shirt_img[center_y - patch_size//2:center_y + patch_size//2,
                         center_x - patch_size//2:center_x + patch_size//2]

# Apply bilateral filtering for smoother texture
fabric_patch = cv2.bilateralFilter(fabric_patch, d=9, sigmaColor=75, sigmaSpace=75)

# Save high-quality fabric patch
cv2.imwrite("fabric_patch_high_quality.jpg", fabric_patch)

print("High-quality fabric patch extracted successfully!")
