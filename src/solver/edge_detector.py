"""
Edge detection logic for piece analysis.
Detects straight edges, measures straightness, and calculates alignment rotations.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from src.utils.puzzle_piece import CornerData, EdgeData


# Edge detection thresholds
MIN_EDGE_LENGTH = 15
MIN_EDGE_STRAIGHTNESS = 0.75
MIN_EDGE_SCORE = 0.3


def detect_edges(mask: np.ndarray,
                 piece_center: Tuple[float, float],
                 corner_data_list: List[CornerData]) -> List[EdgeData]:
    """
    Detect straight edges on the piece (excluding corner edges).

    Returns:
        List of EdgeData objects, sorted by quality
    """
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)

    # Approximate to get segments
    epsilon = 0.008 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n = len(approx)

    if n < 3:
        return []

    # Get corner positions for filtering
    corner_positions = {c.position for c in corner_data_list}

    # Calculate piece dimensions
    piece_h, piece_w = mask.shape
    piece_perimeter = cv2.arcLength(contour, True)

    edge_data_list = []

    # Check each edge segment
    for i in range(n):
        p1 = tuple(approx[i][0])
        p2 = tuple(approx[(i + 1) % n][0])

        # Skip if this edge connects two corner points
        if p1 in corner_positions and p2 in corner_positions:
            continue

        # Calculate edge properties
        edge_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        length = float(np.linalg.norm(edge_vector))

        if length < MIN_EDGE_LENGTH:
            continue

        # Find segment in original contour for straightness measurement
        idx1 = _find_point_in_contour(contour, p1)
        idx2 = _find_point_in_contour(contour, p2)

        if idx1 == -1 or idx2 == -1:
            continue

        # Measure straightness
        straightness = _measure_edge_straightness(contour, idx1, idx2)

        if straightness < MIN_EDGE_STRAIGHTNESS:
            continue

        # Calculate edge angle
        edge_angle = float(np.degrees(np.arctan2(edge_vector[1], edge_vector[0])))
        edge_angle = edge_angle % 360

        # Calculate edge direction (unit vector)
        edge_direction = edge_vector / length

        # Calculate inward normal
        cx, cy = piece_center
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        perp1 = np.array([-edge_direction[1], edge_direction[0]])
        perp2 = np.array([edge_direction[1], -edge_direction[0]])

        to_center = np.array([cx - mid_x, cy - mid_y])
        if np.dot(perp1, to_center) > np.dot(perp2, to_center):
            inward_normal = perp1
        else:
            inward_normal = perp2

        # Score
        length_score = min(1.0, length / (piece_perimeter * 0.2))
        overall_quality = (
            0.6 * straightness +
            0.4 * length_score
        )

        if overall_quality < MIN_EDGE_SCORE:
            continue

        # Calculate rotations to align to each direction
        rotations_to_align = {
            'right': _calculate_rotation_to_align(edge_angle, 0),
            'bottom': _calculate_rotation_to_align(edge_angle, 90),
            'left': _calculate_rotation_to_align(edge_angle, 180),
            'top': _calculate_rotation_to_align(edge_angle, 270),
        }

        edge_data = EdgeData(
            start_point=p1,
            end_point=p2,
            midpoint=(int(mid_x), int(mid_y)),
            length=length,
            straightness=straightness,
            angle=edge_angle,
            quality=overall_quality,
            rotations_to_align=rotations_to_align
        )

        edge_data_list.append(edge_data)

    # Sort by quality
    edge_data_list.sort(key=lambda e: e.quality, reverse=True)

    return edge_data_list[:2]  # Return top 2


def calculate_edge_rotations(edge_data_list: List[EdgeData]) -> Dict[str, List[float]]:
    """Calculate rotations to align edges to each cardinal direction."""
    edge_rotations = {'bottom': [], 'right': [], 'top': [], 'left': []}

    for edge_data in edge_data_list:
        for direction, rotation in edge_data.rotations_to_align.items():
            edge_rotations[direction].append(rotation)

    return edge_rotations


def _calculate_rotation_to_align(current_angle: float, target_angle: float) -> float:
    """Calculate shortest rotation to align current_angle to target_angle."""
    diff = (target_angle - current_angle) % 360
    if diff > 180:
        diff -= 360
    return float(-diff)


def _find_point_in_contour(contour: np.ndarray, point: tuple) -> int:
    """Find the index of a point in the contour."""
    for i, pt in enumerate(contour):
        if tuple(pt[0]) == point:
            return i
    return -1


def _measure_edge_straightness(contour: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Measure how straight an edge is (0-1, where 1.0 is perfectly straight)."""
    n = len(contour)

    if end_idx < start_idx:
        end_idx += n

    indices = [(i % n) for i in range(start_idx, end_idx + 1)]
    edge_points = contour[indices]

    if len(edge_points) < 3:
        return 1.0

    p_start = edge_points[0][0]
    p_end = edge_points[-1][0]

    straight_dist = float(np.linalg.norm(p_end - p_start))

    if straight_dist < 1:
        return 1.0

    path_dist = 0.0
    for i in range(len(edge_points) - 1):
        p1 = edge_points[i][0]
        p2 = edge_points[i + 1][0]
        path_dist += float(np.linalg.norm(p2 - p1))

    straightness = float(straight_dist / path_dist if path_dist > 0 else 0.0)

    return min(1.0, straightness)
