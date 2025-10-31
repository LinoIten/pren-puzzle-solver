# src/ui/simulator/guess_renderer.py

import numpy as np
import cv2
from typing import List, Dict


class GuessRenderer:
    def __init__(self, width: int = 1000, height: int = 1000):
        self.width = width
        self.height = height
        # Define colors for different pieces (BGR format)
        self.piece_colors = [
            (255, 100, 100),  # Blue-ish
            (100, 255, 100),  # Green-ish
            (100, 100, 255),  # Red-ish
            (255, 255, 100),  # Cyan-ish
            (255, 100, 255),  # Magenta-ish
            (100, 255, 255),  # Yellow-ish
        ]
    
    def render(self, guess: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Render a guess onto a canvas.
        Returns grayscale for scoring, but we'll make a color version for visualization.
        """
        canvas = np.zeros((self.height, self.width), dtype=np.float32)
        
        for placement in guess:
            piece_id = placement['piece_id']
            x, y = int(placement['x']), int(placement['y'])
            theta = placement['theta']
            
            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]
                
                # Rotate shape
                rotated = self._rotate_shape(shape, theta)
                
                # Place on canvas
                self._place_shape(canvas, rotated, x, y, value=1.0)
        
        return canvas
    
    def render_color(self, guess: List[dict], piece_shapes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Render a guess with different colors for each piece (for visualization).
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for placement in guess:
            piece_id = placement['piece_id']
            x, y = int(placement['x']), int(placement['y'])
            theta = placement['theta']
            
            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]
                
                # Rotate shape
                rotated = self._rotate_shape(shape, theta)
                
                # Get color for this piece
                color = self.piece_colors[piece_id % len(self.piece_colors)]
                
                # Place on canvas with color
                self._place_shape_color(canvas, rotated, x, y, color)
        
        return canvas
    
    def _rotate_shape(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees."""
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
        return rotated
    
    def _place_shape(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, value: float = 1.0):
        """Place shape on canvas, incrementing pixel values."""
        h, w = shape.shape[:2]
        
        # Calculate bounds
        y1 = max(0, y - h // 2)
        y2 = min(canvas.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(canvas.shape[1], x + w // 2)
        
        shape_y1 = max(0, h // 2 - y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, w // 2 - x)
        shape_x2 = shape_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            canvas[y1:y2, x1:x2] += shape[shape_y1:shape_y2, shape_x1:shape_x2] * value
    
    def _place_shape_color(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, color: tuple):
        """Place colored shape on canvas."""
        h, w = shape.shape[:2]
        
        # Calculate bounds
        y1 = max(0, y - h // 2)
        y2 = min(canvas.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(canvas.shape[1], x + w // 2)
        
        shape_y1 = max(0, h // 2 - y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, w // 2 - x)
        shape_x2 = shape_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            shape_region = shape[shape_y1:shape_y2, shape_x1:shape_x2]
            mask = shape_region > 0
            
            for c in range(3):
                canvas[y1:y2, x1:x2, c][mask] = color[c]
                