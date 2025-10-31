from typing import List
import numpy as np
import cv2

class GuessRenderer:
    def __init__(self, width: int = 1000, height: int = 1000):
        self.width = width
        self.height = height
    
    def render(self, guess: List[dict], piece_shapes: dict) -> np.ndarray:
        """
        Render a guess onto a canvas.
        
        Args:
            guess: List of placements
            piece_shapes: Dict mapping piece_id -> contour/mask
            
        Returns:
            Canvas with rendered pieces (grayscale, counts overlaps)
        """
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for placement in guess:
            piece_id = placement['piece_id']
            x, y = int(placement['x']), int(placement['y'])
            theta = placement['theta']
            
            # Get piece shape (you'll need to pass this in)
            if piece_id in piece_shapes:
                shape = piece_shapes[piece_id]
                
                # Rotate shape
                rotated = self._rotate_shape(shape, theta)
                
                # Place on canvas at (x, y)
                self._place_shape(canvas, rotated, x, y)
        
        return canvas
    
    def _rotate_shape(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees."""
        h, w = shape.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(shape, M, (w, h))
        return rotated
    
    def _place_shape(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int):
        """Place shape on canvas, incrementing pixel values (for overlap detection)."""
        h, w = shape.shape[:2]
        
        # Calculate bounds
        y1 = max(0, y - h // 2)
        y2 = min(canvas.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(canvas.shape[1], x + w // 2)
        
        # Add shape to canvas (increment for overlap counting)
        canvas[y1:y2, x1:x2] += shape[:y2-y1, :x2-x1]