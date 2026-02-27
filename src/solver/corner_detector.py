"""
Corner detection logic for piece analysis.
Detects 90-degree corners and calculates overhang penalties.
"""

import cv2
import numpy as np
from typing import List, Tuple

from src.utils.puzzle_piece import CornerData


# Corner detection thresholds
ANGLE_TOL = 6
MIN_CORNER_STRAIGHTNESS = 0.9
MIN_CORNER_EDGE_LENGTH = 25


def detect_corners(mask: np.ndarray,
                   piece_center: Tuple[float, float]) -> List[CornerData]:
    """
    Detect 90-degree corners on the piece.

    Returns:
        List of CornerData objects, sorted by quality
    """
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)

    # Approximate to reduce noise
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n = len(approx)

    if n < 3:
        return []

    cx, cy = piece_center
    piece_h, piece_w = mask.shape
    piece_perimeter = cv2.arcLength(contour, True)
    typical_edge_length = piece_perimeter / 6
    piece_size_ref = min(piece_h, piece_w)

    corner_data_list = []

    # Check each point for corner
    for i in range(n):
        p_prev = approx[(i - 1) % n][0]
        p_curr = approx[i][0]
        p_next = approx[(i + 1) % n][0]

        # Calculate vectors
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        v1_len = np.linalg.norm(v1)
        v2_len = np.linalg.norm(v2)

        # Skip if edges too short
        if v1_len < MIN_CORNER_EDGE_LENGTH or v2_len < MIN_CORNER_EDGE_LENGTH:
            continue

        # Normalize vectors
        v1n = v1 / v1_len
        v2n = v2 / v2_len

        # Calculate angle
        dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(dot)))

        # Check if close to 90 degrees
        angle_error = abs(90 - angle)
        if angle_error > ANGLE_TOL:
            continue

        angle_score = 1.0 - (angle_error / ANGLE_TOL)

        # Measure straightness (simplified)
        edge1_straightness = 0.9
        edge2_straightness = 0.9

        if edge1_straightness < MIN_CORNER_STRAIGHTNESS or edge2_straightness < MIN_CORNER_STRAIGHTNESS:
            continue

        # Calculate distance from center
        distance_from_center = np.sqrt((p_curr[0] - cx)**2 + (p_curr[1] - cy)**2)
        max_distance = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2) / 2
        distance_score = distance_from_center / max_distance

        # Edge length scoring
        min_edge_len = min(v1_len, v2_len)
        avg_edge_len = (v1_len + v2_len) / 2

        perimeter_score = min(1.0, avg_edge_len / (typical_edge_length * 0.6))
        dimension_score = min(1.0, avg_edge_len / (piece_size_ref * 0.4))
        min_edge_score = min(1.0, min_edge_len / (typical_edge_length * 0.4))

        edge_length_score = (
            0.40 * min_edge_score +
            0.35 * perimeter_score +
            0.25 * dimension_score
        )

        # Overall quality score
        straightness_avg = (edge1_straightness + edge2_straightness) / 2
        overall_quality = (
            0.45 * edge_length_score +
            0.30 * straightness_avg +
            0.15 * angle_score +
            0.10 * distance_score
        )

        # Calculate bisector angle
        v1_unit = v1 / v1_len
        v2_unit = v2 / v2_len
        bisector = (v1_unit + v2_unit) / 2
        bisector_angle = np.degrees(np.arctan2(bisector[1], bisector[0]))
        bisector_angle = bisector_angle % 360

        # Calculate rotation to align corner to bottom-right
        target_bisector = 135.0
        raw_rotation = (target_bisector - bisector_angle) % 360
        rotation_to_align = -(raw_rotation + 90)

        # Would this corner cause the piece to extend beyond puzzle boundaries?
        overhang_penalty = calculate_corner_overhang(
            mask, p_curr, rotation_to_align, piece_center
        )

        # Reduce quality if corner would cause overhang
        quality_before = overall_quality
        overall_quality = overall_quality * (1.0 - overhang_penalty)

        # DEBUG: Show quality reduction
        if overhang_penalty > 0.1:
            print(f"      ⚠️  Corner quality: {quality_before:.3f} → {overall_quality:.3f} (penalty={overhang_penalty:.3f})")

        if overall_quality < 0.65:
            continue

        corner_data = CornerData(
            position=(int(p_curr[0]), int(p_curr[1])),
            angle=float(angle),
            quality=float(overall_quality),
            edge_lengths=(float(v1_len), float(v2_len)),
            rotation_to_align=float(rotation_to_align)
        )

        corner_data_list.append(corner_data)

    # Sort by quality
    corner_data_list.sort(key=lambda c: c.quality, reverse=True)

    # DEBUG: Print corner summary
    if len(corner_data_list) > 0:
        print(f"    📍 Detected {len(corner_data_list)} corner(s):")
        for i, corner in enumerate(corner_data_list[:4]):
            print(f"       #{i+1}: pos={corner.position}, quality={corner.quality:.3f}, rot={corner.rotation_to_align:.1f}°")

    return corner_data_list[:4]  # Return top 4


def calculate_corner_overhang(mask: np.ndarray,
                              corner_point: np.ndarray,
                              rotation: float,
                              piece_center: Tuple[float, float]) -> float:
    """
    Calculate how much a piece would overhang if this corner is placed at puzzle corner.

    This actually rotates the piece and checks if it fits in a corner region.

    Returns:
        Overhang penalty (0.0 = no overhang, 1.0 = severe overhang)
    """
    # Get piece dimensions
    piece_h, piece_w = mask.shape
    cx, cy = piece_center
    corner_x, corner_y = int(corner_point[0]), int(corner_point[1])

    # STEP 1: Rotate the piece to align the corner
    if abs(rotation) > 0.1:
        center = (piece_w // 2, piece_h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((piece_h * sin) + (piece_w * cos))
        new_h = int((piece_h * cos) + (piece_w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))

        corner_homogeneous = np.array([corner_x, corner_y, 1])
        rotated_corner = M @ corner_homogeneous
        corner_x_rot = int(rotated_corner[0])
        corner_y_rot = int(rotated_corner[1])
    else:
        rotated_mask = mask
        corner_x_rot = corner_x
        corner_y_rot = corner_y
        new_w = piece_w
        new_h = piece_h

    # STEP 2: Check how much extends beyond corner placement
    piece_points = np.argwhere(rotated_mask > 0)
    if len(piece_points) == 0:
        return 0.0

    min_y, min_x = piece_points.min(axis=0)
    max_y, max_x = piece_points.max(axis=0)

    extends_left = corner_x_rot - min_x
    extends_right = max_x - corner_x_rot
    extends_up = corner_y_rot - min_y
    extends_down = max_y - corner_y_rot

    max_allowed_overhang = 20

    right_overhang = max(0, extends_right - max_allowed_overhang)
    down_overhang = max(0, extends_down - max_allowed_overhang)

    min_required_extent = 60

    left_deficit = max(0, min_required_extent - extends_left)
    up_deficit = max(0, min_required_extent - extends_up)

    piece_size = max(new_w, new_h)

    overhang_penalty = (right_overhang + down_overhang) / piece_size
    extent_penalty = (left_deficit + up_deficit) / piece_size

    total_penalty = overhang_penalty + extent_penalty * 0.5

    if total_penalty > 0.1:
        print(f"      [CHECK] Overhang check at ({corner_x}, {corner_y}), rot={rotation:.1f}:")
        print(f"         Extends: L={extends_left}px, R={extends_right}px, U={extends_up}px, D={extends_down}px")
        print(f"         Overhang: right={right_overhang}px, down={down_overhang}px")
        print(f"         Deficit: left={left_deficit}px, up={up_deficit}px")
        print(f"         Penalties: overhang={overhang_penalty:.3f}, extent={extent_penalty:.3f}")
        print(f"         TOTAL PENALTY: {total_penalty:.3f}")

    return min(1.0, total_penalty)
