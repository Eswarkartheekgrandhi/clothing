import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image_path, num_colors=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    hex_colors = []
    for color in dominant_colors:
        hex_color = "#{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])
        hex_colors.append(hex_color)
    return hex_colors

def create_prompt(hex_colors):
    prompt = f"Fabric with a pattern, dominant colors: {', '.join(hex_colors)}, detailed texture"  
    return prompt

if __name__ == "__main__":
    image_path = "shirt.jpg"  
    hex_colors = extract_colors(image_path)
    prompt = create_prompt(hex_colors)

    print(f"Dominant Colors: {hex_colors}")
    print(f"Prompt: {prompt}")
    with open("prompt.txt", "w") as f:
        f.write(prompt)