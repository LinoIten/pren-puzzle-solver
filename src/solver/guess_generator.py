# src/solver/guess_generator.py

from itertools import permutations, product
from typing import Sequence, Tuple, List
import numpy as np


class GuessGenerator:
    """Generates candidate placements for puzzle pieces."""
    
    def __init__(self, rotation_step: int = 90):
        self.rotation_step = rotation_step
    
    # In guess_generator.py

    def generate_grid_positions(self, 
                                target: np.ndarray, 
                                grid_spacing: int = 40) -> List[Tuple[float, float]]:
        """
        Generate a grid of positions across the target area.
        
        Args:
            target: Target layout mask
            grid_spacing: Distance between grid points in pixels
            
        Returns:
            List of (x, y) positions to try
        """
        # Find target bounding box
        y_coords, x_coords = np.where(target > 0)
        
        if len(x_coords) == 0:
            # Fallback if no target
            return [(400.0, 400.0)]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Expand slightly beyond target
        margin = 150
        x_min = max(0, x_min - margin)
        x_max = min(target.shape[1], x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(target.shape[0], y_max + margin)
        
        # Generate grid
        positions = []
        for y in range(int(y_min), int(y_max), grid_spacing):
            for x in range(int(x_min), int(x_max), grid_spacing):
                positions.append((float(x), float(y)))
        
        return positions
    
    def generate_guesses(self, 
                        num_pieces: int, 
                        target: np.ndarray,
                        max_guesses: int = 10000) -> List[List[dict]]:
        """
        Generate guesses where each piece can move to different positions.
        
        Args:
            num_pieces: Number of puzzle pieces
            target: Target layout to determine search area
            max_guesses: Maximum number of guesses to generate
            
        Returns:
            List of guesses, each guess is a list of {piece_id, x, y, theta}
        """
        # Generate grid of candidate positions
        positions = self.generate_grid_positions(target, grid_spacing=80)
        
        # Get rotations
        rotations = list(range(0, 360, self.rotation_step))
        piece_ids = list(range(num_pieces))
        
        print(f"Generated {len(positions)} grid positions")
        print(f"Rotations: {len(rotations)}")
        print(f"Pieces: {num_pieces}")
        
        all_guesses = []
        
        # For each piece ordering
        for piece_order in permutations(piece_ids):
            # For each rotation combination
            for rotation_combo in product(rotations, repeat=num_pieces):
                # For each position combination (this is the key change!)
                # Sample positions to avoid explosion
                import random
                sampled_positions = random.sample(positions, min(len(positions), 6))
                
                for position_combo in product(sampled_positions, repeat=num_pieces):
                    guess = []
                    for piece_id, pos, theta in zip(piece_order, position_combo, rotation_combo):
                        guess.append({
                            'piece_id': piece_id,
                            'x': pos[0],
                            'y': pos[1],
                            'theta': theta
                        })
                    all_guesses.append(guess)
                    
                    # Limit total guesses
                    if len(all_guesses) >= max_guesses:
                        print(f"Reached max guesses limit: {max_guesses}")
                        return all_guesses
        
        print(f"Generated {len(all_guesses)} total guesses")
        return all_guesses
    
    