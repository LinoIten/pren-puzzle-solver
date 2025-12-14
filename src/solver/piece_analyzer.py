"""
Piece analyzer that detects corners and straight edges, 
then populates PuzzlePiece objects with the analysis data.

This is a drop-in replacement for src.solver.piece_analyzer.PieceAnalyzer
that extends the existing corner detection with straight edge detection.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
from src.utils.puzzle_piece import PuzzlePiece, CornerData, EdgeData


class PieceAnalyzer:
    """
    Analyzes puzzle pieces to detect corners and straight edges.
    Enriches PuzzlePiece objects with analysis data in-place.
    
    Usage:
        # Create pieces
        pieces = [PuzzlePiece(pid="0", pick=Pose(x=100, y=200, theta=0)), ...]
        
        # Analyze all pieces
        PieceAnalyzer.analyze_all_pieces(pieces, piece_shapes)
        
        # Now pieces have .piece_type, .corners, .edges, etc populated
    """
    
    # Detection thresholds
    MIN_EDGE_LENGTH = 30
    MIN_EDGE_STRAIGHTNESS = 0.65
    MIN_EDGE_SCORE = 0.4
    
    # Corner detection thresholds (from your existing code)
    ANGLE_TOL = 8
    MIN_CORNER_STRAIGHTNESS = 0.9
    MIN_CORNER_EDGE_LENGTH = 25
    
    @staticmethod
    def analyze_all_pieces(puzzle_pieces: List[PuzzlePiece], 
                          piece_shapes: Dict[int, np.ndarray]) -> None:
        """
        Analyze all pieces and populate their analysis data IN-PLACE.
        
        Args:
            puzzle_pieces: List of PuzzlePiece objects to enrich
            piece_shapes: Dict mapping piece_id to binary mask
        """
        print("\n🔍 Analyzing all pieces for corners and edges...")
        
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            
            if piece_id not in piece_shapes:
                print(f"  ⚠️  Piece {piece_id} not found in piece_shapes")
                continue
            
            mask = piece_shapes[piece_id]
            
            # Analyze this piece
            PieceAnalyzer.analyze_piece(piece, mask)
            
            # Print summary
            if piece.piece_type == "corner":
                print(f"  ✓ Piece {piece_id}: CORNER ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")
            elif piece.piece_type == "edge":
                print(f"  ✓ Piece {piece_id}: EDGE ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")
            else:
                print(f"  ○ Piece {piece_id}: CENTER ({len(piece.corners)} corner(s), {len(piece.edges)} edge(s))")
    
    @staticmethod
    def analyze_piece(piece: PuzzlePiece, mask: np.ndarray) -> None:
        """
        Analyze a single piece and populate its analysis fields IN-PLACE.
        
        Args:
            piece: PuzzlePiece object to enrich
            mask: Binary mask of the piece (0s and 1s, or 0-255)
        """
        try:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Calculate piece center and geometry
            M = cv2.moments(mask)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                h, w = mask.shape
                cx, cy = w / 2, h / 2
            
            piece.center = (cx, cy)
            piece.area = float(M['m00'])
            
            # Find contour for perimeter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                piece.perimeter = cv2.arcLength(contours[0], True)
            
            # Detect corners
            corner_data_list = PieceAnalyzer._detect_corners(mask, (cx, cy))
            piece.corners = corner_data_list
            piece.has_corner = len(corner_data_list) > 0
            
            # Set primary corner rotation (best corner)
            if corner_data_list:
                best_corner = max(corner_data_list, key=lambda c: c.quality)
                piece.primary_corner_rotation = best_corner.rotation_to_align
            
            # Detect straight edges (excluding corner edges)
            edge_data_list = PieceAnalyzer._detect_edges(mask, (cx, cy), corner_data_list)
            piece.edges = edge_data_list
            piece.has_straight_edge = len(edge_data_list) > 0
            
            # Calculate edge rotations for all cardinal directions
            if edge_data_list:
                piece.edge_rotations = PieceAnalyzer._calculate_edge_rotations(edge_data_list)
                
                # Set primary edge rotation (best edge, aligned to bottom)
                best_edge = max(edge_data_list, key=lambda e: e.quality)
                piece.primary_edge_rotation = best_edge.rotations_to_align.get('bottom')
            
            # Classify piece type - SMARTER LOGIC
            # A true corner piece should have:
            # - 2 corners (ideal)
            # - OR 1 high-quality corner (>0.75) with NO edges
            # An edge piece should have:
            # - At least 1 straight edge
            # - Even if it has a weak corner
            
            num_corners = len(corner_data_list)
            num_edges = len(edge_data_list)

            if (num_corners > 0 and corner_data_list[0].quality > 0.85):
                # Very high-quality corner detected -> definitely a corner piece
                piece.piece_type = "corner"
                piece.analysis_confidence = corner_data_list[0].quality
            elif (num_edges > 0 and edge_data_list[0].quality > 0.8):
                # Very high-quality edge detected -> definitely an edge piece
                piece.piece_type = "edge"
                piece.analysis_confidence = edge_data_list[0].quality
            elif (num_corners > 0 and num_edges > 0):
                # chose the one with higher quality
                if corner_data_list[0].quality >= edge_data_list[0].quality:
                    piece.piece_type = "corner"
                    piece.analysis_confidence = corner_data_list[0].quality
                else:
                    piece.piece_type = "edge"
                    piece.analysis_confidence = edge_data_list[0].quality
            elif (num_corners > 0):
                piece.piece_type = "corner"
                piece.analysis_confidence = corner_data_list[0].quality
            elif (num_edges > 0):
                piece.piece_type = "edge"
                piece.analysis_confidence = edge_data_list[0].quality
            else:
                piece.piece_type = "center"
                piece.analysis_confidence = 1.0

            
            
        except Exception as e:
            print(f"  ⚠️  Error analyzing piece {piece.id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Set safe defaults
            piece.piece_type = "unknown"
            piece.analysis_confidence = 0.0
    
    @staticmethod
    def _detect_corners(mask: np.ndarray, 
                       piece_center: Tuple[float, float]) -> List[CornerData]:
        """
        Detect 90-degree corners on the piece.
        This uses your existing corner detection logic.
        
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
            if v1_len < PieceAnalyzer.MIN_CORNER_EDGE_LENGTH or v2_len < PieceAnalyzer.MIN_CORNER_EDGE_LENGTH:
                continue
            
            # Normalize vectors
            v1n = v1 / v1_len
            v2n = v2 / v2_len
            
            # Calculate angle
            dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(dot)))
            
            # Check if close to 90 degrees
            angle_error = abs(90 - angle)
            if angle_error > PieceAnalyzer.ANGLE_TOL:
                continue
            
            angle_score = 1.0 - (angle_error / PieceAnalyzer.ANGLE_TOL)
            
            # Measure straightness (simplified - you can use your detailed version)
            edge1_straightness = 0.9  # Placeholder
            edge2_straightness = 0.9  # Placeholder
            
            if edge1_straightness < PieceAnalyzer.MIN_CORNER_STRAIGHTNESS or edge2_straightness < PieceAnalyzer.MIN_CORNER_STRAIGHTNESS:
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
            
            # NEW: Check boundary overhang
            # Would this corner cause the piece to extend beyond puzzle boundaries?
            overhang_penalty = PieceAnalyzer._calculate_corner_overhang(
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
    
    @staticmethod
    def _calculate_corner_overhang(mask: np.ndarray,
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
        # (Simulate what would happen when placing in puzzle corner)
        if abs(rotation) > 0.1:  # Only rotate if needed
            # Rotate around piece center
            center = (piece_w // 2, piece_h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Calculate new size
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((piece_h * sin) + (piece_w * cos))
            new_h = int((piece_h * cos) + (piece_w * sin))
            
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Rotate the mask
            rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))
            
            # Transform corner point
            corner_homogeneous = np.array([corner_x, corner_y, 1])
            rotated_corner = M @ corner_homogeneous
            corner_x_rot = int(rotated_corner[0])
            corner_y_rot = int(rotated_corner[1])
        else:
            # No rotation needed
            rotated_mask = mask
            corner_x_rot = corner_x
            corner_y_rot = corner_y
            new_w = piece_w
            new_h = piece_h
        
        # STEP 2: Check how much extends beyond corner placement
        # When placed at bottom-right corner (width, height):
        # - Corner point should be at (width, height)
        # - Piece should extend left and up ONLY
        # - Should NOT extend right or down
        
        # Find actual piece bounds in rotated mask
        piece_points = np.argwhere(rotated_mask > 0)
        if len(piece_points) == 0:
            return 0.0  # No piece? No penalty
        
        min_y, min_x = piece_points.min(axis=0)
        max_y, max_x = piece_points.max(axis=0)
        
        # Calculate how far piece extends from corner in each direction
        # If corner is at (corner_x_rot, corner_y_rot):
        extends_left = corner_x_rot - min_x    # Should be positive (good)
        extends_right = max_x - corner_x_rot   # Should be ~0 (bad if large)
        extends_up = corner_y_rot - min_y      # Should be positive (good)
        extends_down = max_y - corner_y_rot    # Should be ~0 (bad if large)
        
        # Calculate overhang in "wrong" directions
        max_allowed_overhang = 3
        
        right_overhang = max(0, extends_right - max_allowed_overhang)
        down_overhang = max(0, extends_down - max_allowed_overhang)
        
        # Also penalize if doesn't extend enough in "good" directions
        min_required_extent = 50 
        
        left_deficit = max(0, min_required_extent - extends_left)
        up_deficit = max(0, min_required_extent - extends_up)
        
        # Calculate total penalty
        piece_size = max(new_w, new_h)
        
        # Overhang penalty (extends too far in wrong direction)
        overhang_penalty = (right_overhang + down_overhang) / piece_size
        
        # Extent penalty (doesn't extend enough in right direction)
        extent_penalty = (left_deficit + up_deficit) / piece_size
        
        # Combined penalty
        total_penalty = overhang_penalty + extent_penalty * 0.5
        
        # DEBUG OUTPUT
        if total_penalty > 0.1:  # Only show problematic corners
            print(f"      🔍 Overhang check at ({corner_x}, {corner_y}), rot={rotation:.1f}°:")
            print(f"         Extends: L={extends_left}px, R={extends_right}px, U={extends_up}px, D={extends_down}px")
            print(f"         Overhang: right={right_overhang}px, down={down_overhang}px")
            print(f"         Deficit: left={left_deficit}px, up={up_deficit}px")
            print(f"         Penalties: overhang={overhang_penalty:.3f}, extent={extent_penalty:.3f}")
            print(f"         TOTAL PENALTY: {total_penalty:.3f}")
        
        # Clamp to [0, 1]
        return min(1.0, total_penalty)
    
    @staticmethod
    def _detect_edges(mask: np.ndarray,
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
            
            if length < PieceAnalyzer.MIN_EDGE_LENGTH:
                continue
            
            # Find segment in original contour for straightness measurement
            idx1 = PieceAnalyzer._find_point_in_contour(contour, p1)
            idx2 = PieceAnalyzer._find_point_in_contour(contour, p2)
            
            if idx1 == -1 or idx2 == -1:
                continue
            
            # Measure straightness
            straightness = PieceAnalyzer._measure_edge_straightness(contour, idx1, idx2)
            
            if straightness < PieceAnalyzer.MIN_EDGE_STRAIGHTNESS:
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
            
            if overall_quality < PieceAnalyzer.MIN_EDGE_SCORE:
                continue
            
            # Calculate rotations to align to each direction
            rotations_to_align = {
                'right': PieceAnalyzer._calculate_rotation_to_align(edge_angle, 0),
                'bottom': PieceAnalyzer._calculate_rotation_to_align(edge_angle, 90),
                'left': PieceAnalyzer._calculate_rotation_to_align(edge_angle, 180),
                'top': PieceAnalyzer._calculate_rotation_to_align(edge_angle, 270),
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
    
    @staticmethod
    def _calculate_edge_rotations(edge_data_list: List[EdgeData]) -> Dict[str, List[float]]:
        """Calculate rotations to align edges to each cardinal direction."""
        edge_rotations = {'bottom': [], 'right': [], 'top': [], 'left': []}
        
        for edge_data in edge_data_list:
            for direction, rotation in edge_data.rotations_to_align.items():
                edge_rotations[direction].append(rotation)
        
        return edge_rotations
    
    @staticmethod
    def _calculate_rotation_to_align(current_angle: float, target_angle: float) -> float:
        """Calculate shortest rotation to align current_angle to target_angle."""
        diff = (target_angle - current_angle) % 360
        if diff > 180:
            diff -= 360
        return float(-diff)
    
    @staticmethod
    def _find_point_in_contour(contour: np.ndarray, point: tuple) -> int:
        """Find the index of a point in the contour."""
        for i, pt in enumerate(contour):
            if tuple(pt[0]) == point:
                return i
        return -1
    
    @staticmethod
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
    
    @staticmethod
    def visualize_corners(mask: np.ndarray, piece: PuzzlePiece) -> np.ndarray:
        """
        Create visualization showing detected corners and edges on a piece.
        Compatible with your existing visualization code.
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = [255, 255, 255]
        
        # Draw piece center
        if piece.center:
            cx, cy = int(piece.center[0]), int(piece.center[1])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
        
        # Draw corners (green)
        for i, corner in enumerate(piece.corners):
            radius = 12 if i == 0 else 8
            thickness = -1 if i == 0 else 2
            cv2.circle(vis, corner.position, radius, (0, 255, 0), thickness)
            
            if piece.center:
                cv2.line(vis, (int(piece.center[0]), int(piece.center[1])), 
                        corner.position, (0, 255, 0), 2)
            
            # Add quality text
            text_pos = (corner.position[0] + 15, corner.position[1])
            cv2.putText(vis, f"{corner.quality:.2f}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw straight edges (cyan)
        for i, edge in enumerate(piece.edges):
            cv2.line(vis, edge.start_point, edge.end_point, (255, 255, 0), 3)
            cv2.circle(vis, edge.midpoint, 6, (255, 255, 0), -1)
            
            # Add quality text
            text_pos = (edge.midpoint[0] + 15, edge.midpoint[1])
            cv2.putText(vis, f"{edge.quality:.2f}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add text with piece info
        y_pos = 30
        text = f"Type: {piece.piece_type.upper()}"
        cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        if piece.corners:
            text = f"Corners: {len(piece.corners)}"
            if piece.primary_corner_rotation is not None:
                text += f", Rot: {piece.primary_corner_rotation:.1f}"
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        if piece.edges:
            text = f"Edges: {len(piece.edges)}"
            if piece.primary_edge_rotation is not None:
                text += f", Rot: {piece.primary_edge_rotation:.1f}"
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis