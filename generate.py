import cv2
import colorgram
import numpy as np
from sklearn.cluster import KMeans
import random

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

def generate_random_shirt_shape(height, width):
    """Generates a random shirt-like shape."""
    # Create a blank image
    shirt = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a rectangle for the body of the shirt
    body_top_left = (width // 4, height // 4)
    body_bottom_right = (3 * width // 4, 3 * height // 4)
    cv2.rectangle(shirt, body_top_left, body_bottom_right, (255, 255, 255), -1)
    
    # Draw sleeves
    sleeve_width = width // 8
    sleeve_height = height // 4
    left_sleeve_top_left = (0, height // 3)
    left_sleeve_bottom_right = (width // 4, 2 * height // 3)
    right_sleeve_top_left = (3 * width // 4, height // 3)
    right_sleeve_bottom_right = (width, 2 * height // 3)
    cv2.rectangle(shirt, left_sleeve_top_left, left_sleeve_bottom_right, (255, 255, 255), -1)
    cv2.rectangle(shirt, right_sleeve_top_left, right_sleeve_bottom_right, (255, 255, 255), -1)
    
    # Draw a collar
    collar_points = np.array([
        [width // 3, height // 4],
        [2 * width // 3, height // 4],
        [width // 2, height // 6]
    ], dtype=np.int32)
    cv2.fillPoly(shirt, [collar_points], (255, 255, 255))
    
    return shirt

def apply_colors_and_pattern_to_shirt(shirt, dominant_colors, pattern, alpha=0.5):
    """Applies dominant colors and pattern to the shirt."""
    # Resize pattern to match shirt dimensions
    pattern_resized = cv2.resize(pattern, (shirt.shape[1], shirt.shape[0]))
    pattern_resized = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
    
    # Apply dominant colors to the shirt
    shirt_pixels = shirt.reshape((-1, 3))
    kmeans = KMeans(n_clusters=len(dominant_colors))
    kmeans.fit(shirt_pixels)
    labels = kmeans.labels_.reshape(shirt.shape[:2])
    
    for i in range(shirt.shape[0]):
        for j in range(shirt.shape[1]):
            region_label = labels[i, j]
            shirt[i, j] = [dominant_colors[region_label].b, dominant_colors[region_label].g, dominant_colors[region_label].r]
    
    # Overlay the pattern with transparency
    result = cv2.addWeighted(shirt, 1 - alpha, pattern_resized, alpha, 0)
    return result

def generate_shirt_designs(cloth_image_path, num_designs=5):
    """Generates random shirt designs with extracted colors and patterns."""
    try:
        # Extract dominant colors from the cloth image
        colors = colorgram.extract(cloth_image_path, 5)  # Adjust number of colors as needed
        dominant_colors = [color.rgb for color in colors]
        
        # Extract pattern from the cloth image
        pattern = extract_pattern(cloth_image_path)
        
        for design_num in range(num_designs):
            # Generate a random shirt shape
            height, width = 500, 400  # Adjust dimensions as needed
            shirt = generate_random_shirt_shape(height, width)
            
            # Apply colors and pattern to the shirt
            result = apply_colors_and_pattern_to_shirt(shirt, dominant_colors, pattern)
            
            # Yield the generated shirt design
            yield result

    except Exception as e:
        print(f"An error occurred: {e}")

# Path to the cloth image
cloth_image_path = "shirt.jpg"

# Generate and save shirt designs
for i, shirt_design in enumerate(generate_shirt_designs(cloth_image_path, num_designs=5)):
    output_path = f"random_shirt_design_{i+1}.jpg"
    cv2.imwrite(output_path, shirt_design)
    print(f"Shirt design saved to: {output_path}")