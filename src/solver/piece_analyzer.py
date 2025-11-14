from multiprocessing.util import debug
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CornerQuality:
    """Quality metrics for a detected corner."""
    position: Tuple[int, int]
    angle: float  # Actual angle in degrees
    angle_score: float  # How close to 90 degrees (1.0 = perfect)
    edge1_straightness: float  # How straight the first edge is (0-1)
    edge2_straightness: float  # How straight the second edge is (0-1)
    edge_lengths: Tuple[float, float]  # Lengths of adjacent edges
    bisector_angle: float  # Angle of corner bisector (direction corner "points")
    overall_score: float  # Combined quality score


@dataclass
class PieceCornerInfo:
    """Information about a piece's corner characteristics."""
    piece_id: int
    has_corner: bool
    corner_count: int
    corner_positions: List[Tuple[int, int]]
    corner_qualities: List[CornerQuality]  # NEW: Quality info for each corner
    primary_corner_angle: Optional[float]
    rotation_to_bottom_right: Optional[float]
    piece_center: Tuple[float, float]


class PieceAnalyzer:
    """Analyze puzzle pieces using OpenCV corner detection."""
    
    @staticmethod
    def analyze_piece(piece_id: int, mask: np.ndarray) -> PieceCornerInfo:
        """Analyze a piece to detect corners and determine orientation."""
        try:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Find piece center
            M = cv2.moments(mask)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                h, w = mask.shape
                cx, cy = w / 2, h / 2
            
            piece_center = (cx, cy)
            
            # Detect corners with quality metrics
            corner_qualities = PieceAnalyzer._detect_corners_with_quality(mask, piece_center)
            
            # Extract just the positions
            corners = [cq.position for cq in corner_qualities]
            
            has_corner = len(corners) >= 1
            corner_count = len(corners)
            
            primary_corner_angle = None
            rotation_to_bottom_right = None
            
            if has_corner and corner_qualities:
                # Select PRIMARY corner based on quality, not just distance
                # Prioritize: 90° accuracy > straightness > distance from center
                primary_corner_quality = max(corner_qualities, key=lambda cq: cq.overall_score)
                primary_corner = primary_corner_quality.position
                
                print(f"    Piece {piece_id} primary corner quality:")
                print(f"      Angle: {primary_corner_quality.angle:.1f}° (score: {primary_corner_quality.angle_score:.3f})")
                print(f"      Edge straightness: {primary_corner_quality.edge1_straightness:.3f}, {primary_corner_quality.edge2_straightness:.3f}")
                print(f"      Bisector points at: {primary_corner_quality.bisector_angle:.1f}°")
                print(f"      Overall score: {primary_corner_quality.overall_score:.3f}")
                
                # For a bottom-right corner in a rectangular target:
                # The vectors v1 and v2 point FROM the corner TO adjacent points (inward)
                # So the bisector points INWARD toward the piece center
                # For a bottom-right corner, the inward bisector points at 135° (northwest)

                bisector_angle = primary_corner_quality.bisector_angle
                target_bisector = 135.0  # Bottom-right corner's inward bisector

                # Calculate raw rotation needed
                raw_rotation = (target_bisector - bisector_angle) % 360

                # Apply the transformation that the renderer expects
                # This is the theta value we'll use directly in guesses
                rotation_to_bottom_right = -(raw_rotation + 90)
                
                # Also store the position-based angle for reference
                dx = primary_corner[0] - cx
                dy = primary_corner[1] - cy
                primary_corner_angle = np.degrees(np.arctan2(dy, dx))
            
            return PieceCornerInfo(
                piece_id=piece_id,
                has_corner=has_corner,
                corner_count=corner_count,
                corner_positions=corners,
                corner_qualities=corner_qualities,
                primary_corner_angle=primary_corner_angle,
                rotation_to_bottom_right=rotation_to_bottom_right,
                piece_center=piece_center
            )
        except Exception as e:
            print(f"  ⚠️  Error analyzing piece {piece_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return safe defaults
            return PieceCornerInfo(
                piece_id=piece_id,
                has_corner=False,
                corner_count=0,
                corner_positions=[],
                corner_qualities=[],
                primary_corner_angle=None,
                rotation_to_bottom_right=None,
                piece_center=(0.0, 0.0)
            )
        
    @staticmethod
    def _measure_edge_straightness(contour: np.ndarray, start_idx: int, end_idx: int) -> float:
        """
        Measure how straight an edge is between two contour points.
        Returns 0-1, where 1.0 is perfectly straight.
        """
        n = len(contour)
        
        # Handle wrapping
        if end_idx < start_idx:
            end_idx += n
        
        # Get all points along this edge
        indices = [(i % n) for i in range(start_idx, end_idx + 1)]
        edge_points = contour[indices]
        
        if len(edge_points) < 3:
            return 1.0  # Too short to measure, assume straight
        
        # Get start and end points
        p_start = edge_points[0][0]
        p_end = edge_points[-1][0]
        
        # Calculate ideal straight line distance
        straight_dist = float(np.linalg.norm(p_end - p_start))
        
        if straight_dist < 1:
            return 1.0
        
        # Calculate actual path length
        path_dist = 0.0
        for i in range(len(edge_points) - 1):
            p1 = edge_points[i][0]
            p2 = edge_points[i + 1][0]
            path_dist += float(np.linalg.norm(p2 - p1))
        
        # Straightness = straight_dist / path_dist
        # If path is curved, path_dist > straight_dist
        straightness = float(straight_dist / path_dist if path_dist > 0 else 0.0)
        
        return min(1.0, straightness)
    
    @staticmethod
    def _detect_corners_with_quality(mask: np.ndarray, 
                                     piece_center: Tuple[float, float]) -> List[CornerQuality]:
        """
        Robust 90-degree corner detection with quality metrics.
        Returns corners sorted by quality.
        """
        # Ensure uint8 binary mask
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255

        # Find largest contour (piece outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)

        # Approximate to reduce noise but keep detail
        epsilon = 0.01 * cv2.arcLength(contour, True)  # Reduced from 0.015
        approx = cv2.approxPolyDP(contour, epsilon, True)
        n = len(approx)
        if n < 3:
            return []

        cx, cy = piece_center
        
        corner_qualities = []
        
        # Parameters for corner detection
        ANGLE_TOL = 10  # Must be within 10 degrees of 90°
        MIN_STRAIGHTNESS = 0.85  # Edges must be at least 85% straight
        MIN_EDGE_LENGTH = 15  # Minimum edge length in pixels
        
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
            if v1_len < MIN_EDGE_LENGTH or v2_len < MIN_EDGE_LENGTH:
                continue

            # Normalize vectors
            v1n = v1 / v1_len
            v2n = v2 / v2_len
            
            # Calculate angle
            dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(dot)))  # Use abs to handle both convex and concave
            
            # Check if close to 90 degrees
            angle_error = abs(90 - angle)
            if angle_error > ANGLE_TOL:
                continue
            
            # Score how close to 90 degrees (1.0 = perfect)
            angle_score = 1.0 - (angle_error / ANGLE_TOL)
            
            # Measure straightness of adjacent edges
            # We need to find the actual contour segments, not just the approximated points
            idx_curr = -1
            for j, pt in enumerate(contour):
                if np.array_equal(pt[0], p_curr):
                    idx_curr = j
                    break
            
            if idx_curr == -1:
                continue  # Couldn't find point in original contour
            
            # Find neighboring points in original contour
            # Go back along contour to find previous approximated point
            idx_prev = -1
            for k in range(1, len(contour)):
                test_idx = (idx_curr - k) % len(contour)
                if np.array_equal(contour[test_idx][0], p_prev):
                    idx_prev = test_idx
                    break
            
            # Go forward along contour to find next approximated point
            idx_next = -1
            for k in range(1, len(contour)):
                test_idx = (idx_curr + k) % len(contour)
                if np.array_equal(contour[test_idx][0], p_next):
                    idx_next = test_idx
                    break
            
            if idx_prev == -1 or idx_next == -1:
                # Fallback: use simple distance-based straightness
                edge1_straightness = 0.9
                edge2_straightness = 0.9
            else:
                # Measure straightness of both edges
                edge1_straightness = PieceAnalyzer._measure_edge_straightness(contour, idx_prev, idx_curr)
                edge2_straightness = PieceAnalyzer._measure_edge_straightness(contour, idx_curr, idx_next)
            
            # Filter by straightness
            if edge1_straightness < MIN_STRAIGHTNESS or edge2_straightness < MIN_STRAIGHTNESS:
                continue
            
            # Calculate distance from center (normalized)
            distance_from_center = np.sqrt((p_curr[0] - cx)**2 + (p_curr[1] - cy)**2)
            max_distance = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2) / 2
            distance_score = distance_from_center / max_distance
            
            # Calculate bisector angle - the direction the corner "points"
            # This is the angle that bisects the two edges forming the corner
            # Get unit vectors for both edges (pointing AWAY from corner)
            v1_unit = v1 / v1_len
            v2_unit = v2 / v2_len
            
            # Bisector is the average of the two unit vectors
            bisector = (v1_unit + v2_unit) / 2
            bisector_angle = np.degrees(np.arctan2(bisector[1], bisector[0]))
            
            # Normalize to 0-360
            bisector_angle = bisector_angle % 360
            
            # Calculate overall quality score
            # Prioritize: angle accuracy (40%) > straightness (40%) > distance (20%)
            straightness_avg = (edge1_straightness + edge2_straightness) / 2
            overall_score = (
                0.4 * angle_score +
                0.4 * straightness_avg +
                0.2 * distance_score
            )
            
            corner_quality = CornerQuality(
                position=(int(p_curr[0]), int(p_curr[1])),  # Explicit tuple of 2 ints
                angle=float(angle),
                angle_score=float(angle_score),
                edge1_straightness=float(edge1_straightness),
                edge2_straightness=float(edge2_straightness),
                edge_lengths=(float(v1_len), float(v2_len)),  # Explicit tuple of 2 floats
                bisector_angle=float(bisector_angle),
                overall_score=float(overall_score)
            )
            
            corner_qualities.append(corner_quality)
        
        # Sort by overall quality score (best first)
        corner_qualities.sort(key=lambda cq: cq.overall_score, reverse=True)
        
        # Limit to 4 best corners
        return corner_qualities[:4]


    @staticmethod
    def analyze_all_pieces(piece_shapes: Dict[int, np.ndarray]) -> Dict[int, PieceCornerInfo]:
        """Analyze all pieces and return corner information."""
        results = {}
        
        print("\n🔍 Analyzing all pieces for corners...")
        
        for piece_id, mask in piece_shapes.items():
            info = PieceAnalyzer.analyze_piece(piece_id, mask)
            results[piece_id] = info
            
            if info.has_corner:
                print(f"  ✓ Piece {piece_id}: {info.corner_count} corner(s) detected")
                if info.primary_corner_angle is not None:
                    print(f"    Primary corner at {info.primary_corner_angle:.1f}°")
                    print(f"    Needs {info.rotation_to_bottom_right:.1f}° rotation to align bottom-right")
            else:
                print(f"  ○ Piece {piece_id}: No corners detected (edge/center piece)")
        
        return results
    
    @staticmethod
    def visualize_corners(mask: np.ndarray, 
                        corner_info: PieceCornerInfo) -> np.ndarray:
        """Create visualization showing detected corners with quality indicators."""
        # Ensure mask is proper format
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Create a colored version of the mask
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = [255, 255, 255]  # White piece
        
        # Draw piece center
        cx, cy = int(corner_info.piece_center[0]), int(corner_info.piece_center[1])
        cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)  # Blue center
        
        # Draw all detected corners with quality-based colors
        for i, (corner, quality) in enumerate(zip(corner_info.corner_positions, corner_info.corner_qualities)):
            # Color based on quality: green (best) -> yellow -> red (worst)
            quality_score = quality.overall_score
            if quality_score > 0.8:
                color = (0, 255, 0)  # Green - excellent
            elif quality_score > 0.6:
                color = (0, 255, 255)  # Yellow - good
            else:
                color = (0, 165, 255)  # Orange - acceptable
            
            # Primary corner gets bigger circle
            radius = 12 if i == 0 else 8
            thickness = -1 if i == 0 else 2
            
            cv2.circle(vis, corner, radius, color, thickness)
            
            # Draw line from center to corner
            cv2.line(vis, (cx, cy), corner, color, 2)
            
            # Add quality text near corner
            text_pos = (corner[0] + 15, corner[1])
            cv2.putText(vis, f"{quality_score:.2f}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add text with rotation info
        y_pos = 30
        if corner_info.rotation_to_bottom_right is not None:
            text = f"Rotate: {corner_info.rotation_to_bottom_right:.1f}"
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
            y_pos += 30
        
        text2 = f"Corners: {corner_info.corner_count}"
        cv2.putText(vis, text2, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
        
        # Show quality of primary corner if available
        if corner_info.corner_qualities:
            primary = corner_info.corner_qualities[0]
            y_pos += 30
            text3 = f"Angle: {primary.angle:.1f}deg"
            cv2.putText(vis, text3, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1)
            y_pos += 25
            text4 = f"Straight: {primary.edge1_straightness:.2f}, {primary.edge2_straightness:.2f}"
            cv2.putText(vis, text4, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1)
        
        return vis