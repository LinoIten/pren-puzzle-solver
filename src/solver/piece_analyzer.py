from multiprocessing.util import debug
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PieceCornerInfo:
    """Information about a piece's corner characteristics."""
    piece_id: int
    has_corner: bool
    corner_count: int
    corner_positions: List[Tuple[int, int]]
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
            
            # Use contour-based corner detection (more reliable for puzzle pieces)
            corners = PieceAnalyzer._detect_corners_from_contour(mask)
            
            has_corner = len(corners) >= 1
            corner_count = len(corners)
            
            primary_corner_angle = None
            rotation_to_bottom_right = None
            
            if has_corner and corners:
                # Get the primary corner (furthest from center)
                primary_corner = max(corners, key=lambda c: 
                    np.sqrt((c[0] - cx)**2 + (c[1] - cy)**2))
                
                # Calculate angle from piece center to corner
                dx = primary_corner[0] - cx
                dy = primary_corner[1] - cy
                primary_corner_angle = np.degrees(np.arctan2(dy, dx))
                
                # Calculate rotation needed to move corner to bottom-right
                # Bottom-right is at 315 degrees (or -45) from center
                target_angle = 315
                rotation_to_bottom_right = (target_angle - primary_corner_angle) % 360
            
            return PieceCornerInfo(
                piece_id=piece_id,
                has_corner=has_corner,
                corner_count=corner_count,
                corner_positions=corners,
                primary_corner_angle=primary_corner_angle,
                rotation_to_bottom_right=rotation_to_bottom_right,
                piece_center=piece_center
            )
        except Exception as e:
            print(f"  ⚠️  Error analyzing piece {piece_id}: {e}")
            # Return safe defaults
            return PieceCornerInfo(
                piece_id=piece_id,
                has_corner=False,
                corner_count=0,
                corner_positions=[],
                primary_corner_angle=None,
                rotation_to_bottom_right=None,
                piece_center=(0.0, 0.0)
            )
    
    @staticmethod
    def _detect_corners_from_contour(mask: np.ndarray) -> list[tuple[int, int]]:
        """
        Robust 90-degree corner detection from a binary mask.
        Works for rotated and noisy puzzle pieces.
        """
        # Ensure uint8 binary mask
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255

        # 1️⃣ Find largest contour (piece outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)

        # 2️⃣ Approximate the contour to simplify (reduce noise)
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        n = len(approx)
        if n < 3:
            return []

        # 3️⃣ Calculate piece center
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            h, w = mask.shape
            cx, cy = w / 2, h / 2

        # 4️⃣ Filter by local 90° geometry and straightness
        corners = []
        ANGLE_TOL = 12       # angle tolerance in degrees (adjust if needed)
        STRAIGHT_TOL = 0.98  # how straight the adjacent edges must be (1 = perfect line)

        for i in range(n):
            p1 = approx[(i - 1) % n][0]
            p2 = approx[i][0]
            p3 = approx[(i + 1) % n][0]

            v1 = p1 - p2
            v2 = p3 - p2
            v1_len = np.linalg.norm(v1)
            v2_len = np.linalg.norm(v2)
            if v1_len < 5 or v2_len < 5:
                continue

            # Normalize
            v1n = v1 / v1_len
            v2n = v2 / v2_len
            dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))

            # 90° ± tolerance
            if 90 - ANGLE_TOL <= angle <= 90 + ANGLE_TOL:
                # Check that both adjacent segments are fairly straight
                seg1 = cv2.approxPolyDP(np.array([p1, p2]), 1, False)
                seg2 = cv2.approxPolyDP(np.array([p2, p3]), 1, False)

                if len(seg1) == 2 and len(seg2) == 2:  # both are lines
                    corners.append(tuple(map(int, p2)))

        # 5️⃣ Optional: limit to 4 strongest corners (farthest from center)
        if len(corners) > 4:
            corners = sorted(
                corners,
                key=lambda c: np.hypot(c[0] - cx, c[1] - cy),
                reverse=True
            )[:4]

        return corners


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
        """Create visualization showing detected corners."""
        # Ensure mask is proper format
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Create a colored version of the mask
        # Show the actual piece shape in white, background in black
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = [255, 255, 255]  # White piece
        
        # Draw piece center
        cx, cy = int(corner_info.piece_center[0]), int(corner_info.piece_center[1])
        cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)  # Blue center
        
        # Draw all detected corners
        for i, corner in enumerate(corner_info.corner_positions):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for primary, red for others
            cv2.circle(vis, corner, 8, color, -1)
            
            # Draw line from center to corner
            cv2.line(vis, (cx, cy), corner, color, 2)
        
        # Add text with rotation info
        if corner_info.rotation_to_bottom_right is not None:
            text = f"Rotate: {corner_info.rotation_to_bottom_right:.1f}"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        text2 = f"Corners: {corner_info.corner_count}"
        cv2.putText(vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
        
        return vis