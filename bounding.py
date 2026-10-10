# Bounding Box Coordinate Transformation:Write a Python function that converts absolute pixel xyxy coordinates 3637 for an $800 \times 600$ image into normalized xywh format.
import numpy as np
import cv2


def convert_xyxy_to_xywh(xyxy, image_width, image_height):
    """
    Convert absolute pixel xyxy coordinates to normalized xywh format.

    Parameters:
    xyxy (list or np.array): A list or array containing the bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    image_width (int): The width of the image.
    image_height (int): The height of the image.

    Returns:
    np.array: A numpy array containing the normalized bounding box coordinates in the format [x_center, y_center, width, height].
    """
    # Ensure xyxy is a numpy array
    xyxy = np.array(xyxy)

    # calculate the width and height of the bounding box
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]

    # calculate the center coordinates of the bounding box
    x_center = xyxy[0] + width / 2
    y_center = xyxy[1] + height / 2

    # normalize the coordinates
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return np.array([x_center_normalized, y_center_normalized, width_normalized, height_normalized])

# Example usage
if __name__ == "__main__":
    # Absolute pixel coordinates for a bounding box
    xyxy = [100, 150, 300, 400]  # [x_min, y_min, x_max, y_max]
    image_width = 800
    image_height = 600

    # Convert to normalized xywh format
    normalized_xywh = convert_xyxy_to_xywh(xyxy, image_width, image_height)
    print("Normalized xywh coordinates:", normalized_xywh)