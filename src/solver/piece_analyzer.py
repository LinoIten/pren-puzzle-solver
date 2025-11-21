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
    edge_length_score: float  # Score based on edge lengths (0-1)
    bisector_angle: float  # Angle of corner bisector (direction corner "points")
    overall_score: float  # Combined quality score



@dataclass
class PieceCornerInfo:
    """Information about a piece's corner characteristics."""
    piece_id: int
    has_corner: bool
    corner_count: int
    corner_positions: List[Tuple[int, int]]
    corner_qualities: List[CornerQuality]
    corner_rotations: List[float]  
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
            corner_rotations = []
            
            # Calculate rotation for ALL detected corners WITH fine-tuning
            for i, corner_quality in enumerate(corner_qualities):
                bisector_angle = corner_quality.bisector_angle
                target_bisector = 135.0  # Bottom-right corner's inward bisector
                
                # Calculate initial raw rotation
                raw_rotation = (target_bisector - bisector_angle) % 360
                initial_rotation = -(raw_rotation + 90)
                
                # FINE-TUNE the rotation
                print(f"      Corner {i+1}: quality={corner_quality.overall_score:.3f}, edge_score={corner_quality.edge_length_score:.3f}, initial={initial_rotation:.1f}°")
                refined_rotation = PieceAnalyzer._fine_tune_corner_rotation(
                    piece_id,
                    mask,
                    initial_rotation,
                    search_range=3.0,  # Search ±3 degrees
                    step=0.15  # 0.15 degree steps 
                )
                
                corner_rotations.append(refined_rotation)
            
            if has_corner and corner_qualities:
                # Select PRIMARY corner based on quality
                primary_corner_quality = max(corner_qualities, key=lambda cq: cq.overall_score)
                primary_corner = primary_corner_quality.position
                
                print(f"    Piece {piece_id} detected {len(corner_qualities)} corner(s):")
                for i, (cq, rot) in enumerate(zip(corner_qualities, corner_rotations)):
                    is_primary = cq == primary_corner_quality
                    marker = "⭐" if is_primary else "  "
                    print(f"      {marker} Corner {i+1}: quality={cq.overall_score:.3f}, edges={cq.edge_lengths[0]:.1f}/{cq.edge_lengths[1]:.1f}px, rotation={rot:.1f}°")
                
                # Store primary corner rotation (the one with highest quality)
                primary_idx = corner_qualities.index(primary_corner_quality)
                rotation_to_bottom_right = corner_rotations[primary_idx]
                
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
                corner_rotations=corner_rotations,  # NEW: All rotations
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
                corner_rotations=[],  # NEW
                primary_corner_angle=None,
                rotation_to_bottom_right=None,
                piece_center=(0.0, 0.0)
            )
        

    @staticmethod
    def _fine_tune_corner_rotation(piece_id: int,
                                mask: np.ndarray,
                                initial_rotation: float,
                                search_range: float = 5.0,
                                step: float = 0.25) -> float:
        """
        Fine-tune rotation by testing angles around initial estimate.
        Returns the rotation that best fits a bottom-right corner.
        """
        # Get piece dimensions
        piece_h, piece_w = mask.shape
        
        # Create test canvas proportional to piece size
        # Make it 2x the piece size to give room for rotation
        test_size = int(max(piece_h, piece_w) * 2)
        
        target = np.zeros((test_size, test_size), dtype=np.uint8)
        target[:, :] = 1  # Fill entire area
        
        best_rotation = initial_rotation
        best_score = -float('inf')
        
        print(f"        Fine-tuning: test canvas {test_size}x{test_size}, piece {piece_w}x{piece_h}")
        
        # Test rotations around initial estimate
        for angle_offset in np.arange(-search_range, search_range + step, step):
            test_rotation = float(initial_rotation + angle_offset)
            
            # Rotate and crop piece
            rotated = PieceAnalyzer._rotate_and_crop_piece(mask, test_rotation)
            if rotated is None:
                continue
            
            rot_h, rot_w = rotated.shape
            
            # Place piece in bottom-right corner of test canvas
            x = test_size - rot_w
            y = test_size - rot_h
            
            # Check if piece fits
            if x < 0 or y < 0:
                continue
            
            # Render piece on test canvas
            test_canvas = np.zeros((test_size, test_size), dtype=np.float32)
            test_canvas[y:y+rot_h, x:x+rot_w] = rotated
            
            # Score based on corner alignment
            # The piece's bottom-right corner should align perfectly with canvas corner
            piece_br_x = x + rot_w - 1
            piece_br_y = y + rot_h - 1
            canvas_br_x = test_size - 1
            canvas_br_y = test_size - 1
            
            # Distance from perfect alignment (lower is better)
            x_error = abs(piece_br_x - canvas_br_x)
            y_error = abs(piece_br_y - canvas_br_y)
            alignment_error = x_error + y_error
            
            # Also check edge alignment - edges should be flush with canvas edges
            # Check bottom edge: how many pixels are in the bottom row?
            bottom_edge_coverage = np.sum(test_canvas[-1, :] > 0) / test_size
            # Check right edge: how many pixels are in the right column?
            right_edge_coverage = np.sum(test_canvas[:, -1] > 0) / test_size
            
            # Combined score (minimize alignment error, maximize edge coverage)
            score = -alignment_error + (bottom_edge_coverage + right_edge_coverage) * 100
            
            if score > best_score:
                best_score = score
                best_rotation = test_rotation
        
        # Calculate improvement
        improvement = abs(best_rotation - initial_rotation)
        
        if improvement > 0.1:  # More than 0.1 degree change
            print(f"        Refined: {initial_rotation:.2f}° → {best_rotation:.2f}° (Δ{improvement:.2f}°)")
            return float(best_rotation)
        else:
            return float(initial_rotation)

    @staticmethod
    def _rotate_and_crop_piece(mask: np.ndarray, angle: float) -> Optional[np.ndarray]:
        """Rotate and crop a piece mask."""
        if angle == 0:
            return mask.copy()
        
        h, w = mask.shape[:2]
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
        
        rotated = cv2.warpAffine(mask, M, (new_w, new_h))
        
        # Crop to actual content bounds
        piece_points = np.argwhere(rotated > 0)
        if len(piece_points) == 0:
            return None
        
        min_y, min_x = piece_points.min(axis=0)
        max_y, max_x = piece_points.max(axis=0)
        
        cropped = rotated[min_y:max_y+1, min_x:max_x+1]
        return cropped
    
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
        Returns corners sorted by quality, prioritizing long straight edges.
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
        
        # Calculate piece size metrics for dynamic scoring
        piece_h, piece_w = mask.shape
        piece_area = np.sum(mask > 0)
        piece_perimeter = cv2.arcLength(contour, True)
        
        # Estimate "typical" edge length for this piece
        # For a square piece, each edge would be roughly perimeter/4
        # But puzzle pieces are irregular, so use a more conservative estimate
        typical_edge_length = piece_perimeter / 6  # Assume ~6 edges on average
        
        # Also use piece dimensions as reference
        piece_size_ref = min(piece_h, piece_w)  # Smaller dimension
        
        corner_qualities = []
        
        # Parameters for corner detection
        ANGLE_TOL = 10  # Must be within 10 degrees of 90°
        MIN_STRAIGHTNESS = 0.85  # Edges must be at least 85% straight
        MIN_EDGE_LENGTH = 15  # Absolute minimum edge length in pixels
        
        for i in range(n):
            p_prev = approx[(i - 1) % n][0]
            p_curr = approx[i][0]
            p_next = approx[(i + 1) % n][0]

            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            v1_len = np.linalg.norm(v1)
            v2_len = np.linalg.norm(v2)
            
            # Skip if edges too short (absolute minimum)
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
            for j in range(len(contour)):
                if np.array_equal(contour[j][0], p_curr):
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
            
            # === DYNAMIC EDGE LENGTH SCORING ===
            # This is the key improvement: score based on how the edge lengths compare to piece size
            
            # Calculate scores relative to piece-specific metrics
            min_edge_len = min(v1_len, v2_len)
            avg_edge_len = (v1_len + v2_len) / 2
            
            # Score 1: Compare to typical edge length (perimeter-based)
            # A corner edge should be at least 30-40% of typical edge length
            perimeter_score = min(1.0, avg_edge_len / (typical_edge_length * 0.6))
            
            # Score 2: Compare to piece dimensions
            # A corner edge should span a reasonable fraction of piece size
            dimension_score = min(1.0, avg_edge_len / (piece_size_ref * 0.4))
            
            # Score 3: Ensure BOTH edges are reasonably long (not just average)
            # The shorter edge must still be substantial
            min_edge_score = min(1.0, min_edge_len / (typical_edge_length * 0.4))
            
            # Combined edge length score
            # Prioritize: both edges long (40%) > average length (35%) > meets dimension threshold (25%)
            edge_length_score = (
                0.40 * min_edge_score +
                0.35 * perimeter_score +
                0.25 * dimension_score
            )
            
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
            # PRIORITY: edge length (45%) > straightness (30%) > angle accuracy (15%) > distance (10%)
            # Long straight edges are THE most reliable indicator of a puzzle corner
            straightness_avg = (edge1_straightness + edge2_straightness) / 2
            overall_score = (
                0.45 * edge_length_score +
                0.30 * straightness_avg +
                0.15 * angle_score +
                0.10 * distance_score
            )
            
            corner_quality = CornerQuality(
                position=(int(p_curr[0]), int(p_curr[1])),  # Explicit tuple of 2 ints
                angle=float(angle),
                angle_score=float(angle_score),
                edge1_straightness=float(edge1_straightness),
                edge2_straightness=float(edge2_straightness),
                edge_lengths=(float(v1_len), float(v2_len)),  # Explicit tuple of 2 floats
                edge_length_score=float(edge_length_score),  # NEW: edge length score
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
            y_pos += 25
            text5 = f"Edges: {primary.edge_lengths[0]:.0f}px, {primary.edge_lengths[1]:.0f}px"
            cv2.putText(vis, text5, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1)
        
        return vis