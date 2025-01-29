import cv2
import colorgram
import numpy as np
from sklearn.cluster import KMeans

def apply_colors_to_shirt(cloth_image_path, shirt_image_path):
    """Applies dominant colors from a cloth image to a shirt image."""

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

        output_path = "colored_shirt.jpg"
        cv2.imwrite(output_path, shirt_img)
        print(f"Colored shirt saved to: {output_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


cloth_image_path = "shirt.jpg" 
shirt_image_path = "base.jpg" 

apply_colors_to_shirt(cloth_image_path, shirt_image_path)