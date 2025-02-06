import cv2
import colorgram
import numpy as np
from sklearn.cluster import KMeans

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

def apply_colors_and_pattern_to_shirt(cloth_image_path, shirt_image_path):
    """Applies dominant colors and pattern from a cloth image to a shirt image."""

    try:
        shirt_img = cv2.imread(shirt_image_path)
        cloth_img = cv2.imread(cloth_image_path)

        if shirt_img is None:
            raise ValueError(f"Could not load shirt image: {shirt_image_path}")
        if cloth_img is None:
            raise ValueError(f"Could not load cloth image: {cloth_image_path}")

        # 1. Color Extraction
        colors = colorgram.extract(cloth_image_path, 5)  # Adjust number of colors as needed
        dominant_colors = [color.rgb for color in colors]

        # 2. Color Application (K-Means)
        shirt_pixels = shirt_img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters
        kmeans.fit(shirt_pixels)
        labels = kmeans.labels_.reshape(shirt_img.shape[:2])
        centers = kmeans.cluster_centers_.astype(int)

        for i in range(shirt_img.shape[0]):
            for j in range(shirt_img.shape[1]):
                region_label = labels[i, j]
                dominant_color_index = region_label % len(dominant_colors)
                shirt_img[i, j] = [dominant_colors[dominant_color_index].b, dominant_colors[dominant_color_index].g, dominant_colors[dominant_color_index].r]

        # 3. Pattern Extraction and Application
        pattern = extract_pattern(cloth_image_path)
        pattern_resized = cv2.resize(pattern, (shirt_img.shape[1], shirt_img.shape[0]))
        pattern_resized = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
        
        alpha = 0.5  # Adjust the transparency of the pattern
        result = cv2.addWeighted(shirt_img, 1 - alpha, pattern_resized, alpha, 0)

        output_path = "colored_and_patterned_shirt.jpg"
        cv2.imwrite(output_path, result)
        print(f"Colored and patterned shirt saved to: {output_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Paths to the images
cloth_image_path = "shirt.jpg"
shirt_image_path = "base.jpg"

# Apply colors and pattern to the shirt
apply_colors_and_pattern_to_shirt(cloth_image_path, shirt_image_path)