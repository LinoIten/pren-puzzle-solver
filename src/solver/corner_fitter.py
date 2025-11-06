# src/solver/corner_fitter.py

import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class CornerFit:
    """Result of fitting a piece to a corner."""
    piece_id: int
    corner_position: Tuple[float, float]
    rotation: float  # Exact rotation in degrees
    score: float  # How well it fits


class CornerFitter:
    """Precisely fit puzzle pieces to target corners."""
    
    def __init__(self, width: int = 800, height: int = 800):
        self.canvas_width = width
        self.canvas_height = height
    
    
    def score_corner_fit(self, 
                        rendered: np.ndarray, 
                        target: np.ndarray,
                        corner_type: str = 'bottom_right') -> float:
        """
        Score how well a piece fits in a corner.
        
        Returns:
            Score (higher is better, negative if piece goes outside target)
        """
        # Check if piece extends outside target
        outside_target = np.sum((rendered > 0) & (target == 0))
        
        if outside_target > 100:  # Too much outside
            return -1000000.0 - outside_target * 10
        
        # Check coverage inside target
        inside_coverage = np.sum((rendered > 0) & (target > 0))
        
        # Check if piece touches edges
        y_coords, x_coords = np.where(target > 0)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        edge_thickness = 10
        
        # Check edge touching based on corner type
        if corner_type == 'bottom_right':
            touches_bottom = (rendered[y_max - edge_thickness:, :] > 0).any()
            touches_right = (rendered[:, x_max - edge_thickness:] > 0).any()
            touches_edges = touches_bottom and touches_right
        else:
            # Generic edge check
            touches_edges = False
        
        # Calculate score
        score = inside_coverage * 1.0
        
        if touches_edges:
            score += 50000  # Big bonus for touching corner edges
        
        score -= outside_target * 100  # Penalty for going outside
        
        return score
    
    def identify_target_corners(self, target: np.ndarray) -> List[Tuple[float, float, str]]:
        """Identify the 4 corners of target accurately."""
        y_coords, x_coords = np.where(target > 0)
        
        if len(x_coords) == 0:
            print("⚠️  WARNING: Target is empty!")
            return [
                (100.0, 100.0, 'top_left'),
                (700.0, 100.0, 'top_right'),
                (100.0, 700.0, 'bottom_left'),
                (700.0, 700.0, 'bottom_right')
            ]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        target_width = x_max - x_min
        target_height = y_max - y_min
        
        print(f"  📏 Target bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        print(f"  📏 Target size: {target_width}x{target_height}")
        
        corners = [
            (float(x_min), float(y_min), 'top_left'),
            (float(x_max), float(y_min), 'top_right'),
            (float(x_min), float(y_max), 'bottom_left'),
            (float(x_max), float(y_max), 'bottom_right'),
        ]
        
        return corners
        
    def fit_piece_to_corner(self,
                        piece_id: int,
                        piece_mask: np.ndarray,
                        corner_pos: Tuple[float, float],
                        corner_type: str,
                        target: np.ndarray) -> CornerFit:
        """Fit a piece to a corner with exact rotation."""
        # Coarse search
        best_rotation = 0.0
        best_score = -float('inf')
        
        for angle in range(0, 360, 5):
            rotated = self._rotate_mask(piece_mask, float(angle))  # Cast to float
            rendered = self._render_at_position(rotated, corner_pos)
            score = self.score_corner_fit(rendered, target, corner_type)
            
            if score > best_score:
                best_score = score
                best_rotation = float(angle)  # Cast to float
        
        # Fine search around best
        fine_rotation = best_rotation
        fine_score = best_score
        
        for angle in np.arange(best_rotation - 10, best_rotation + 10, 0.5):
            rotated = self._rotate_mask(piece_mask, float(angle))  # Cast to float
            rendered = self._render_at_position(rotated, corner_pos)
            score = self.score_corner_fit(rendered, target, corner_type)
            
            if score > fine_score:
                fine_score = score
                fine_rotation = float(angle)  # Cast to float
        
        return CornerFit(
            piece_id=piece_id,
            corner_position=corner_pos,
            rotation=float(fine_rotation),  # Cast to float
            score=float(fine_score)  # Cast to float
        )

    def _rotate_mask(self, mask: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a mask by exact angle."""
        if angle == 0:
            return mask
        
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(mask, M, (new_w, new_h))
        return rotated
    
    def _render_at_position(self, mask: np.ndarray, pos: Tuple[float, float]) -> np.ndarray:
        """Render a mask at a specific position on the canvas."""
        canvas = np.zeros((self.canvas_height, self.canvas_width), dtype=np.float32)
        
        h, w = mask.shape[:2]
        x, y = int(pos[0]), int(pos[1])
        
        # Calculate bounds
        y1 = max(0, y - h // 2)
        y2 = min(canvas.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(canvas.shape[1], x + w // 2)
        
        mask_y1 = max(0, h // 2 - y)
        mask_y2 = mask_y1 + (y2 - y1)
        mask_x1 = max(0, w // 2 - x)
        mask_x2 = mask_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and mask_y2 > mask_y1 and mask_x2 > mask_x1:
            canvas[y1:y2, x1:x2] = mask[mask_y1:mask_y2, mask_x1:mask_x2]
        
        return canvas