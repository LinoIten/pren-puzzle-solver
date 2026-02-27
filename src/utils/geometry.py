"""
Shared geometry utilities for shape rotation and cropping.
"""

import cv2
import numpy as np


def rotate_and_crop(shape: np.ndarray, angle: float, crop: bool = True) -> np.ndarray:
    """
    Rotate a shape by angle degrees with optional tight cropping.

    Args:
        shape: Binary mask or image to rotate
        angle: Rotation angle in degrees
        crop: If True, crop to tight bounding box of non-zero content

    Returns:
        Rotated (and optionally cropped) array
    """
    if angle == 0:
        return shape

    h, w = shape.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(shape, M, (new_w, new_h))

    if not crop:
        return rotated

    # Crop to actual content bounds
    piece_points = np.argwhere(rotated > 0)
    if len(piece_points) == 0:
        return rotated

    min_y, min_x = piece_points.min(axis=0)
    max_y, max_x = piece_points.max(axis=0)

    return rotated[min_y : max_y + 1, min_x : max_x + 1]
