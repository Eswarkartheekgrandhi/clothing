import cv2
import numpy as np

def extract_patterns_contours(cloth_image_path):
    img_gray = cv2.imread(cloth_image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY) #convert to binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours

    pattern_mask = np.zeros_like(img_gray) #create mask

    for contour in contours:
        # You can add filtering criteria here based on area, shape, etc.
        # Example: Only consider contours with a certain area
        if cv2.contourArea(contour) > 1000:  # Adjust area threshold
            cv2.drawContours(pattern_mask, [contour], -1, (255), -1) #draw contours on mask

    return pattern_mask


cloth_image_path = 'shirt.jpg'  # Replace with your cloth image path
pattern_mask = extract_patterns_contours(cloth_image_path)

cv2.imwrite('pattern_mask.jpg', pattern_mask)
cv2.imshow('Pattern Mask', pattern_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()