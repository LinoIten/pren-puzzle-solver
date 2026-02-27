"""
Cut pattern generation for mock puzzle creation.
Standalone functions for generating wavy, sharp zigzag, and square wave cut lines.
"""

import random

import numpy as np


def generate_wavy_cut(start, end, num_waves=3, amplitude=30):
    """
    Generate a wavy cut line between two points.

    Args:
        start: (x, y) starting point
        end: (x, y) ending point
        num_waves: Number of waves/bumps in the cut
        amplitude: How far the wave deviates from straight line

    Returns:
        Array of points forming the cut line
    """
    x1, y1 = start
    x2, y2 = end

    # Number of points along the line
    num_points = 100

    # Generate base line
    t = np.linspace(0, 1, num_points)
    base_x = x1 + (x2 - x1) * t
    base_y = y1 + (y2 - y1) * t

    # Add wavy perturbation
    wave = amplitude * np.sin(num_waves * 2 * np.pi * t)

    # Calculate perpendicular direction
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        # Perpendicular unit vector
        perp_x = -dy / length
        perp_y = dx / length

        # Add wave perpendicular to the line
        cut_x = base_x + wave * perp_x
        cut_y = base_y + wave * perp_y
    else:
        cut_x = base_x
        cut_y = base_y

    # Convert to integer points
    points = np.column_stack([cut_x, cut_y]).astype(np.int32)

    return points


def generate_sharp_cut(start, end, num_angles=5, amplitude=40):
    """
    Generate a sharp zigzag cut line between two points.

    Args:
        start: (x, y) starting point
        end: (x, y) ending point
        num_angles: Number of sharp angles/zigzags
        amplitude: How far each angle deviates from straight line

    Returns:
        Array of points forming the cut line
    """
    x1, y1 = start
    x2, y2 = end

    # Calculate perpendicular direction
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length
    else:
        perp_x, perp_y = 0, 0

    # Generate points along the line
    points = [start]

    for i in range(1, num_angles + 1):
        # Position along the line
        t = i / (num_angles + 1)
        base_x = x1 + (x2 - x1) * t
        base_y = y1 + (y2 - y1) * t

        # Alternate sides for zigzag effect
        side = 1 if i % 2 == 0 else -1

        # Random amplitude variation for each angle
        current_amp = amplitude * random.uniform(0.7, 1.3)

        # Add perpendicular offset
        point_x = int(base_x + side * current_amp * perp_x)
        point_y = int(base_y + side * current_amp * perp_y)

        points.append((point_x, point_y))

    points.append(end)

    # Convert to numpy array
    return np.array(points, dtype=np.int32)


def generate_square_cut(start, end, num_rectangles=4, amplitude=35):
    """
    Generate a square wave cut line (like a digital signal 0/1).
    Creates rectangular protrusions that stick out from the line.

    Args:
        start: (x, y) starting point
        end: (x, y) ending point
        num_rectangles: Number of rectangular bumps
        amplitude: How far rectangles stick out from the line

    Returns:
        Array of points forming the cut line
    """
    x1, y1 = start
    x2, y2 = end

    # Calculate direction vectors
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        # Unit vectors along and perpendicular to the line
        dir_x = dx / length
        dir_y = dy / length
        perp_x = -dy / length
        perp_y = dx / length
    else:
        return np.array([start, end], dtype=np.int32)

    points = []

    # Width of each rectangle segment along the line
    segment_length = length / (num_rectangles * 2)

    current_pos = 0
    rectangle_index = 0

    # Start at the beginning
    points.append(start)

    while current_pos < length - segment_length:
        # Determine which side this rectangle should be on
        side = 1 if rectangle_index % 2 == 0 else -1

        # Calculate current position on baseline
        base_x = x1 + dir_x * current_pos
        base_y = y1 + dir_y * current_pos

        # Move to next position on baseline
        next_pos = current_pos + segment_length
        next_x = x1 + dir_x * next_pos
        next_y = y1 + dir_y * next_pos

        # Calculate offset perpendicular to line
        offset_x = side * amplitude * perp_x
        offset_y = side * amplitude * perp_y

        # Create rectangle with 90-degree corners:
        # 1. Go perpendicular OUT from baseline
        points.append((int(base_x + offset_x), int(base_y + offset_y)))

        # 2. Move along the offset line (parallel to baseline)
        points.append((int(next_x + offset_x), int(next_y + offset_y)))

        # 3. Go perpendicular back IN to baseline
        points.append((int(next_x), int(next_y)))

        current_pos = next_pos
        rectangle_index += 1

    # End at the end point
    points.append(end)

    return np.array(points, dtype=np.int32)
