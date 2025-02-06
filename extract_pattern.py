import cv2
import numpy as np

def extract_pattern(cloth_image_path):
    """Extracts the pattern from the cloth image."""
    cloth_img = cv2.imread(cloth_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(cloth_img, (5, 5), 0)
    
    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the pattern
    mask = np.zeros_like(cloth_img)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Bitwise-AND to extract the pattern
    pattern = cv2.bitwise_and(cloth_img, cloth_img, mask=mask)
    
    return pattern

def apply_pattern_to_shirt(pattern, shirt_image_path):
    """Applies the extracted pattern to the base shirt."""
    shirt_img = cv2.imread(shirt_image_path)
    
    # Resize the pattern to match the shirt image dimensions
    pattern_resized = cv2.resize(pattern, (shirt_img.shape[1], shirt_img.shape[0]))
    
    # Convert the pattern to a 3-channel image
    pattern_resized = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
    
    # Blend the pattern with the shirt image
    alpha = 0.5  # Adjust the transparency of the pattern
    result = cv2.addWeighted(shirt_img, 1 - alpha, pattern_resized, alpha, 0)
    
    return result

# Paths to the images
cloth_image_path = "shirt.jpg"
shirt_image_path = "base.jpg"

# Extract the pattern from the cloth image
pattern = extract_pattern(cloth_image_path)

# Apply the pattern to the base shirt
result = apply_pattern_to_shirt(pattern, shirt_image_path)

# Save and display the result
output_path = "patterned_shirt.jpg"
cv2.imwrite(output_path, result)
print(f"Patterned shirt saved to: {output_path}")

cv2.imshow('Patterned Shirt', result)
cv2.waitKey(0)
cv2.destroyAllWindows()