import cv2
from PIL import Image

def create_canny_image(image_path, output_path="canny_edges.png"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Optional blur
    edges = cv2.Canny(blurred, 100, 200)  # Adjust thresholds
    canny_image = Image.fromarray(edges)
    canny_image.save(output_path)
    return canny_image

if __name__ == "__main__":
    image_path = "shirt.jpg" 
    canny_image = create_canny_image(image_path)
    print(f"Canny image saved to: canny_edges.png")