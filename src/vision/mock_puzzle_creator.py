# src/vision/mock_puzzle_generator.py

import cv2
import numpy as np
from pathlib import Path
import random


class MockPuzzleGenerator:
    """Generate realistic mock puzzle pieces for testing."""
    
    def __init__(self, output_dir: str = "data/mock_pieces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.a4_width = 420  # Half scale
        self.a4_height = 594
    # Update src/vision/mock_puzzle_creator.py

    def generate_puzzle(self) -> tuple:
        """
        Generate a 4-piece puzzle with slanted wavy cuts.
        
        Returns:
            (full_image, piece_images, debug_image)
        """
        # Create full A4 image (white paper)
        full_image = np.ones((self.a4_height, self.a4_width, 3), dtype=np.uint8) * 255
        
        # Add some texture to make it look like paper
        noise = np.random.randint(-10, 10, (self.a4_height, self.a4_width, 3), dtype=np.int16)
        full_image = np.clip(full_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Define cut positions with randomness AND rotation
        center_x = self.a4_width // 2 + random.randint(-30, 30)
        center_y = self.a4_height // 2 + random.randint(-30, 30)
        
        # Add rotation angles for slanted cuts (in degrees)
        vertical_angle = random.randint(-15, 15)  # Slant the "vertical" cut
        horizontal_angle = random.randint(-15, 15)  # Slant the "horizontal" cut
        
        # Generate vertical cut (with slant)
        # Instead of perfectly vertical, make it angled
        offset_top = int(np.tan(np.radians(vertical_angle)) * self.a4_height / 2)
        vertical_cut = self.generate_wavy_cut(
            (center_x - offset_top, 0),
            (center_x + offset_top, self.a4_height),
            num_waves=4,
            amplitude=random.randint(20, 40)
        )
        
        # Generate horizontal cut (with slant)
        offset_left = int(np.tan(np.radians(horizontal_angle)) * self.a4_width / 2)
        horizontal_cut = self.generate_wavy_cut(
            (0, center_y - offset_left),
            (self.a4_width, center_y + offset_left),
            num_waves=4,
            amplitude=random.randint(20, 40)
        )
        
        # Draw cuts on image for visualization
        debug_image = full_image.copy()
        cv2.polylines(debug_image, [vertical_cut], False, (255, 0, 0), 3)
        cv2.polylines(debug_image, [horizontal_cut], False, (0, 0, 255), 3)
        
        # Create masks for each piece using the actual cut lines
        piece_masks = self._create_piece_masks_from_cuts(vertical_cut, horizontal_cut)
        
        # Extract individual pieces
        piece_images = []
        for i, mask in enumerate(piece_masks):
            # Apply mask to get piece
            piece = cv2.bitwise_and(full_image, full_image, mask=mask)
            
            # Find bounding box
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Add padding
                padding = 5
                x_min = max(0, x_min - padding)
                x_max = min(self.a4_width, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(self.a4_height, y_max + padding)
                
                # Crop piece
                cropped_piece = piece[y_min:y_max+1, x_min:x_max+1]
                cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
                
                piece_images.append({
                    'id': i,
                    'image': cropped_piece,
                    'mask': cropped_mask,
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)
                })
        
        return full_image, piece_images, debug_image

    def _create_piece_masks_from_cuts(self, vertical_cut: np.ndarray, horizontal_cut: np.ndarray) -> list:
        """Create binary masks for each of the 4 pieces using actual cut lines."""
        masks = []
        
        # Create a mask for each quadrant by flood filling
        # First, draw the cuts on a temporary image
        cut_image = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        
        # Draw cuts as lines
        cv2.polylines(cut_image, [vertical_cut], False, 255, 2)
        cv2.polylines(cut_image, [horizontal_cut], False, 255, 2)
        
        # Invert so cuts are black (barriers)
        cut_image = 255 - cut_image
        
        # Find seed points for each quadrant (far from cuts)
        center_x = self.a4_width // 2
        center_y = self.a4_height // 2
        
        # Seed points for 4 quadrants (offset from center to avoid cuts)
        seed_points = [
            (center_x // 2, center_y // 2),  # Top-left
            (center_x + center_x // 2, center_y // 2),  # Top-right
            (center_x // 2, center_y + center_y // 2),  # Bottom-left
            (center_x + center_x // 2, center_y + center_y // 2),  # Bottom-right
        ]
        
        for seed in seed_points:
            # Create mask for this piece using flood fill
            mask = np.zeros((self.a4_height + 2, self.a4_width + 2), dtype=np.uint8)
            temp_image = cut_image.copy()
            
            # Flood fill from seed point
            cv2.floodFill(temp_image, mask, seed, 128)
            
            # Extract the filled region
            piece_mask = (temp_image == 128).astype(np.uint8) * 255
            
            masks.append(piece_mask)
        
        return masks
    
    def generate_wavy_cut(self, 
                         start: tuple, 
                         end: tuple, 
                         num_waves: int = 3,
                         amplitude: int = 30) -> np.ndarray:
        """
        Generate a wavy cut line between two points.
        
        Args:
            start: (x, y) starting point
            end: (x, y) ending point
            num_waves: Number of waves/bumps in the cut
            amplitude: How far the wave deviates from straight line
            
        Returns:
            Array of points forming the cut line
        """
        x1, y1 = start
        x2, y2 = end
        
        # Number of points along the line
        num_points = 100
        
        # Generate base line
        t = np.linspace(0, 1, num_points)
        base_x = x1 + (x2 - x1) * t
        base_y = y1 + (y2 - y1) * t
        
        # Add wavy perturbation
        wave = amplitude * np.sin(num_waves * 2 * np.pi * t)
        
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Add wave perpendicular to the line
            cut_x = base_x + wave * perp_x
            cut_y = base_y + wave * perp_y
        else:
            cut_x = base_x
            cut_y = base_y
        
        # Convert to integer points
        points = np.column_stack([cut_x, cut_y]).astype(np.int32)
        
        return points
    
    def _create_piece_masks(self, vertical_cut: np.ndarray, horizontal_cut: np.ndarray) -> list:
        """Create binary masks for each of the 4 pieces."""
        masks = []
        
        # Create mask for each quadrant
        # Piece 0: Top-left
        mask0 = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        
        # Fill region to the left of vertical cut and above horizontal cut
        for y in range(self.a4_height):
            # Find x position of vertical cut at this y
            vert_indices = np.where(vertical_cut[:, 1] == y)[0]
            if len(vert_indices) > 0:
                x_vert = vertical_cut[vert_indices[0], 0]
            else:
                x_vert = self.a4_width // 2
            
            # Find y position of horizontal cut
            horiz_indices = np.where(horizontal_cut[:, 1] == y)[0]
            if len(horiz_indices) > 0:
                # Only fill if we're above the horizontal cut
                if y < self.a4_height // 2:
                    mask0[y, :x_vert] = 255
        
        # Simpler approach: divide into quadrants
        # This is a simplified version - you can make it more sophisticated
        center_x = self.a4_width // 2
        center_y = self.a4_height // 2
        
        # Top-left
        mask0 = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        mask0[:center_y, :center_x] = 255
        
        # Top-right
        mask1 = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        mask1[:center_y, center_x:] = 255
        
        # Bottom-left
        mask2 = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        mask2[center_y:, :center_x] = 255
        
        # Bottom-right
        mask3 = np.zeros((self.a4_height, self.a4_width), dtype=np.uint8)
        mask3[center_y:, center_x:] = 255
        
        masks = [mask0, mask1, mask2, mask3]
        
        return masks
    
    def save_pieces(self, piece_images: list) -> list:
        """Save piece images to disk and return file paths."""
        saved_paths = []
        
        for piece_data in piece_images:
            piece_id = piece_data['id']
            image = piece_data['image']
            
            # Save as PNG with transparency
            # Create RGBA image
            bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            
            # Set alpha channel from mask
            mask = piece_data['mask']
            bgra[:, :, 3] = mask
            
            # Save
            filepath = self.output_dir / f"piece_{piece_id}.png"
            cv2.imwrite(str(filepath), bgra)
            saved_paths.append(filepath)
            
            print(f"Saved piece {piece_id} to {filepath}")
        
        return saved_paths
    
    def load_pieces_for_solver(self, piece_paths: list = None) -> tuple: # type: ignore
        """
        Load saved pieces and prepare them for the solver.
        
        Returns:
            (piece_ids, piece_shapes_dict)
        """
        if piece_paths is None:
            # Load all pieces from output directory
            piece_paths = sorted(self.output_dir.glob("piece_*.png"))
        
        piece_shapes = {}
        piece_ids = []
        
        for i, path in enumerate(piece_paths):
            # Load image with alpha channel
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                continue
            
            # Extract alpha channel as mask
            if image.shape[2] == 4:
                mask = image[:, :, 3]
            else:
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Normalize to 0 and 1
            mask = (mask > 127).astype(np.uint8)
            
            piece_shapes[i] = mask
            piece_ids.append(i)
        
        return piece_ids, piece_shapes