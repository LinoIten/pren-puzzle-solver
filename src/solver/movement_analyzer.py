# Movement Analysis Utility
# Calculate center of mass for pieces in best solution

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class MovementAnalyzer:
    """Calculate center of mass and movement data for puzzle pieces."""
    
    @staticmethod
    def calculate_piece_com(shape: np.ndarray, x: float, y: float, theta: float) -> Optional[Tuple[float, float]]:
        """
        Calculate center of mass for a piece at given position and rotation.
        
        Args:
            shape: Binary mask of the piece
            x, y: Top-left position of the piece
            theta: Rotation angle in degrees
            
        Returns:
            (com_x, com_y) in absolute coordinates, or None if invalid
        """
        if shape is None or np.sum(shape) == 0:
            return None
        
        # Rotate the shape
        rotated_shape = MovementAnalyzer._rotate_shape(shape, theta)
        
        # Find center of mass of rotated shape
        y_coords, x_coords = np.where(rotated_shape > 0)
        
        if len(x_coords) == 0:
            return None
        
        # COM relative to rotated shape's top-left
        com_x_rel = np.mean(x_coords)
        com_y_rel = np.mean(y_coords)
        
        # Add position offset to get absolute COM
        com_x = com_x_rel + x
        com_y = com_y_rel + y
        
        return (float(com_x), float(com_y))
    
    @staticmethod
    def _rotate_shape(shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees and crop to tight bounding box."""
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
        
        # Crop to actual content bounds
        piece_points = np.argwhere(rotated > 0)
        if len(piece_points) == 0:
            return rotated
        
        min_y, min_x = piece_points.min(axis=0)
        max_y, max_x = piece_points.max(axis=0)
        
        cropped = rotated[min_y:max_y+1, min_x:max_x+1]
        return cropped
    
    @staticmethod
    def analyze_best_solution_movements(
        puzzle_pieces: List, 
        best_guess: List[dict], 
        piece_shapes: Dict[int, np.ndarray],
        surfaces: dict
    ) -> Dict:
        """
        Analyze movements for the best solution.
        
        Returns:
            {
                'source_coms': {piece_id: (x, y)},  # COM in source area
                'target_coms': {piece_id: (x, y)},  # COM in target area  
                'movements': {piece_id: {'distance': float, 'rotation': float}}
            }
        """
        print("\n🔍 Analyzing movements for best solution...")
        
        source_coms = {}
        target_coms = {}
        movements = {}
        
        # Get surface offsets
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']
        target_offset_y = surfaces['target']['offset_y']
        
        # Calculate source COMs (original positions)
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            
            if piece_id in piece_shapes:
                source_com = MovementAnalyzer.calculate_piece_com(
                    piece_shapes[piece_id],
                    piece.pick_pose.x,
                    piece.pick_pose.y,
                    piece.pick_pose.theta
                )
                
                if source_com:
                    # Convert to global coordinates
                    global_source_com = (
                        source_com[0] + source_offset_x,
                        source_com[1] + source_offset_y
                    )
                    source_coms[piece_id] = global_source_com
                    print(f"  Source P{piece_id}: COM at {global_source_com}")
        
        # Calculate target COMs (best solution positions)
        for placement in best_guess:
            piece_id = placement['piece_id']
            
            if piece_id in piece_shapes:
                target_com = MovementAnalyzer.calculate_piece_com(
                    piece_shapes[piece_id],
                    placement['x'],
                    placement['y'],
                    placement['theta']
                )
                
                if target_com:
                    # Convert to global coordinates
                    global_target_com = (
                        target_com[0] + target_offset_x,
                        target_com[1] + target_offset_y
                    )
                    target_coms[piece_id] = global_target_com
                    print(f"  Target P{piece_id}: COM at {global_target_com}")
                    
                    # Calculate movement if we have both source and target
                    if piece_id in source_coms:
                        source = source_coms[piece_id]
                        target = global_target_com
                        
                        # Calculate distance
                        distance = np.sqrt((target[0] - source[0])**2 + (target[1] - source[1])**2)
                        
                        # Calculate rotation change
                        original_piece = next(p for p in puzzle_pieces if int(p.id) == piece_id)
                        rotation_change = (placement['theta'] - original_piece.pick_pose.theta) % 360
                        if rotation_change > 180:
                            rotation_change -= 360  # Use shortest rotation
                        
                        movements[piece_id] = {
                            'distance': float(distance),
                            'rotation': float(rotation_change)
                        }
                        
                        print(f"  Movement P{piece_id}: {distance:.1f}px, {rotation_change:.0f}°")
        
        print(f"✅ Analyzed movements for {len(movements)} pieces")
        
        return {
            'source_coms': source_coms,
            'target_coms': target_coms,
            'movements': movements
        }


def calculate_movement_data_for_visualizer(solution_data):
    """
    Call this in pipeline.py to pre-calculate movement data.
    Add the result to solver_data before launching UI.
    """
    if not solution_data.get('puzzle_pieces') or not solution_data.get('best_guess'):
        print("⚠️  Missing puzzle_pieces or best_guess - skipping movement analysis")
        return None
    
    movement_data = MovementAnalyzer.analyze_best_solution_movements(
        puzzle_pieces=solution_data['puzzle_pieces'],
        best_guess=solution_data['best_guess'],
        piece_shapes=solution_data['piece_shapes'],
        surfaces=solution_data['surfaces']
    )
    
    return movement_data