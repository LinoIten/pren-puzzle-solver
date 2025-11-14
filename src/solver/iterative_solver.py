# src/solver/iterative_solver.py

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass, field

from src.solver.corner_fitter import CornerFit, CornerFitter
from src.solver.piece_analyzer import PieceCornerInfo


@dataclass
class IterativeSolution:
    """Result from iterative solving process."""
    success: bool
    anchor_fit: Optional[CornerFit]
    remaining_placements: List[dict]
    score: float
    iteration: int
    total_iterations: int
    all_guesses: Optional[List[List[dict]]] = None  # NEW: All guesses tried


class IterativeSolver:
    """
    Iterative puzzle solver that tries corner pieces one at a time.
    
    Strategy:
    1. Try each piece that might have a corner
    2. For each piece, try all 4 corners of the target
    3. Find exact rotation for that piece at that corner
    4. Solve remaining pieces
    5. If score is good enough, we're done. Otherwise, try next piece.
    """

    def __init__(self, renderer, scorer, guess_generator):
        self.renderer = renderer
        self.scorer = scorer
        self.guess_generator = guess_generator
        self.corner_fitter = None  # Will be created with target
        self.all_guesses = []
        self.all_scores = []
    
    def solve_iteratively(self,
                         piece_shapes: Dict[int, np.ndarray],
                         piece_corner_info: Dict[int, PieceCornerInfo],
                         target: np.ndarray,
                         score_threshold: float = 50000.0,
                         max_iterations: int = 6) -> IterativeSolution:
        """
        Try pieces iteratively until we find a good solution.
        """
        print("\n🔄 Starting iterative solving...")
        
        # Create corner fitter with target bounds
        target_rows, target_cols = np.where(target > 0)
        width = int(target_cols.max() - target_cols.min())
        height = int(target_rows.max() - target_rows.min())
        self.corner_fitter = CornerFitter(width=width, height=height)
        
        # Reset guess collection
        self.all_guesses = []
        self.all_scores = []
        
        # Get candidates: pieces with corners first, then all others
        corner_pieces = [
            info for info in piece_corner_info.values() 
            if info.has_corner
        ]
        all_pieces = list(piece_corner_info.values())
        
        # Try corner pieces first
        candidates = corner_pieces if corner_pieces else all_pieces
        
        if max_iterations:
            candidates = candidates[:max_iterations]
        
        print(f"  Will try {len(candidates)} candidate pieces as anchors")
        
        best_solution = None
        
        for iteration, candidate in enumerate(candidates):
            print(f"\n--- Iteration {iteration + 1}/{len(candidates)} ---")
            print(f"Trying piece {candidate.piece_id} as anchor")
            
            if candidate.has_corner:
                print(f"  This piece has {candidate.corner_count} detected corner(s)")
            
            # Try this piece as anchor
            solution = self._try_piece_as_anchor(
                candidate,
                piece_shapes,
                target
            )
            
            print(f"✓ Found valid solution with score {solution.score:.1f}")
            
            # Keep best solution
            if best_solution is None or solution.score > best_solution.score:
                best_solution = solution
            
            # If score is good enough, we can stop
            if solution.score >= score_threshold:
                print(f"🎉 Score exceeds threshold ({score_threshold}), stopping!")
                break
        
        if best_solution is None:
            # Return failed solution
            return IterativeSolution(
                success=False,
                anchor_fit=None,
                remaining_placements=[],
                score=-float('inf'),
                iteration=len(candidates),
                total_iterations=len(candidates),
                all_guesses=self.all_guesses  # Include all guesses
            )
        
        print(f"\n🏆 Best solution: score {best_solution.score:.1f}")
        print(f"📊 Total guesses tried: {len(self.all_guesses)}")
        
        # Add all guesses to best solution
        best_solution.all_guesses = self.all_guesses
        
        return best_solution
    
    def _try_piece_as_anchor(self,
                        candidate: PieceCornerInfo,
                        piece_shapes: Dict[int, np.ndarray],
                        target: np.ndarray) -> IterativeSolution:
        """
        Try a specific piece as the anchor corner piece.
        Now uses TOP-LEFT corner positioning for simplicity.
        
        Returns:
            Solution with this piece anchored
        """
        piece_id = candidate.piece_id
        piece_mask = piece_shapes[piece_id]
        
        # Target is already the extracted region
        height, width = target.shape
        
        print(f"\n  === ANCHOR PLACEMENT (TOP-LEFT COORDS) ===")
        print(f"  Target size: {width}x{height}")
        
        # Use the pre-calculated rotation if available
        if candidate.has_corner and candidate.rotation_to_bottom_right is not None:
            initial_rotation = candidate.rotation_to_bottom_right
            print(f"  Pre-calculated rotation: {initial_rotation:.1f}°")
            
            # Rotate and crop the piece
            rotated_mask = self._rotate_and_crop(piece_mask, initial_rotation)
            piece_h, piece_w = rotated_mask.shape
            
            print(f"  Rotated+cropped piece size: {piece_w}x{piece_h}")
            
            # NOW IT'S SIMPLE!
            # We want the piece's BOTTOM-RIGHT corner at target's BOTTOM-RIGHT corner
            # Target BR corner is at (width-1, height-1)
            # Piece BR corner is at (x + piece_w - 1, y + piece_h - 1)
            # So: x + piece_w - 1 = width - 1
            #     x = width - piece_w
            # And: y + piece_h - 1 = height - 1
            #     y = height - piece_h
            
            x = width - piece_w
            y = height - piece_h
            
            print(f"  Placing piece top-left at: ({x}, {y})")
            print(f"  This puts piece BR at: ({x + piece_w - 1}, {y + piece_h - 1})")
            print(f"  Target BR corner is at: ({width - 1}, {height - 1})")
            print(f"  Match: {x + piece_w - 1 == width - 1 and y + piece_h - 1 == height - 1}")
            print(f"  ==========================================\n")
            
            best_fit = CornerFit(
                piece_id=piece_id,
                corner_position=(float(x), float(y)),
                rotation=float(initial_rotation),
                score=1000.0
            )
        else:
            # No corner detected, do full search
            print(f"  No corner detected, doing full rotation search")
            best_fit = self.corner_fitter.fit_piece_to_corner(
                piece_id,
                piece_mask,
                (width - 1, height - 1),
                'bottom_right',
                target
            )
        
        # Now solve for remaining pieces
        remaining_piece_ids = [
            pid for pid in piece_shapes.keys() 
            if pid != piece_id
        ]
        
        print(f"  Solving for {len(remaining_piece_ids)} remaining pieces...")
        
        # Generate guesses for remaining pieces
        guesses = self._generate_guesses_with_anchor(
            best_fit,
            remaining_piece_ids,
            piece_shapes,
            target
        )
        
        print(f"  Testing {len(guesses)} placement combinations...")
        
        if len(guesses) == 0:
            print("  ⚠️  WARNING: No guesses generated!")
            return IterativeSolution(
                success=False,
                anchor_fit=best_fit,
                remaining_placements=[],
                score=-float('inf'),
                iteration=0,
                total_iterations=1,
                all_guesses=[]
            )
        
        # Find best placement for remaining pieces
        best_remaining_score = -float('inf')
        best_guess = None
        
        for i, guess in enumerate(guesses):
            # Store this guess for visualization
            self.all_guesses.append(guess)
            
            rendered = self.renderer.render(guess, piece_shapes)
            score = self.scorer.score(rendered, target)
            
            # Store score too
            self.all_scores.append(score)
            
            if score > best_remaining_score:
                best_remaining_score = score
                best_guess = guess
            
            if i % 500 == 0 and i > 0:
                print(f"    Progress: {i}/{len(guesses)}, best: {best_remaining_score:.1f}")
        
        if best_guess:
            return IterativeSolution(
                success=best_remaining_score > 0,
                anchor_fit=best_fit,
                remaining_placements=best_guess,
                score=best_remaining_score,
                iteration=0,
                total_iterations=1,
                all_guesses=[]
            )
        else:
            return IterativeSolution(
                success=False,
                anchor_fit=best_fit,
                remaining_placements=[],
                score=-float('inf'),
                iteration=0,
                total_iterations=1,
                all_guesses=[]
            )

    def _rotate_and_crop(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate and crop shape - matching what the renderer does."""
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
        
        # Crop to tight bounding box
        cropped = rotated[min_y:max_y+1, min_x:max_x+1]
        
        return cropped

    def _fine_tune_rotation(self,
                        piece_id: int,
                        piece_mask: np.ndarray,
                        corner_pos: Tuple[float, float],
                        initial_rotation: float,
                        target: np.ndarray,
                        search_range: int = 15,
                        step: float = 1.0) -> CornerFit:
        """Fine-tune rotation around the pre-calculated estimate."""

        best_rotation = initial_rotation
        best_score = -float('inf')
        '''
        # Search around initial rotation
        angle = initial_rotation - search_range
        end_angle = initial_rotation + search_range
        
        while angle <= end_angle:
            # Rotate piece
            rotated = self.corner_fitter._rotate_mask(piece_mask, float(angle))  # Cast to float
            
            # Place at corner
            rendered = self.corner_fitter._render_at_position(rotated, corner_pos)
            
            # Score it
            score = self.corner_fitter.score_corner_fit(rendered, target, 'bottom_right')
            
            if score > best_score:
                best_score = score
                best_rotation = float(angle)  # Cast to float
            
            angle += step
        
        print(f"    Fine-tuned: {initial_rotation:.1f}° → {best_rotation:.1f}° (Δ{best_rotation - initial_rotation:.1f}°)")
        '''
        return CornerFit(
            piece_id=piece_id,
            corner_position=corner_pos,
            rotation=float(initial_rotation),  # Cast to float
            score=float(best_score)  # Cast to float
        )

    def _generate_guesses_with_anchor(self,
                                    anchor_fit: CornerFit,
                                    remaining_piece_ids: List[int],
                                    piece_shapes: Dict[int, np.ndarray],
                                    target: np.ndarray,
                                    max_guesses: int = 10) -> List[List[dict]]:
        """
        Generate guesses with one piece fixed as anchor.
        Now uses TOP-LEFT corner positioning.
        """
        import random
        
        print(f"    Generating {max_guesses} guesses for {len(remaining_piece_ids)} pieces...")
        
        # Target IS the canvas
        height, width = target.shape
        
        print(f"    Canvas size: {width}x{height}")
        print(f"    Anchor piece {anchor_fit.piece_id} at ({anchor_fit.corner_position[0]:.0f}, {anchor_fit.corner_position[1]:.0f})")
        
        guesses = []
        
        for guess_num in range(max_guesses):
            guess = []
            
            # Add ANCHORED piece first
            guess.append({
                'piece_id': anchor_fit.piece_id,
                'x': anchor_fit.corner_position[0],
                'y': anchor_fit.corner_position[1],
                'theta': anchor_fit.rotation
            })
            
            # Place all remaining pieces randomly
            for piece_id in remaining_piece_ids:
                # Get piece dimensions at a random rotation
                theta = random.choice([0, 90, 180, 270])
                
                # Rotate piece to get its dimensions
                piece_mask = piece_shapes[piece_id]
                rotated = self._rotate_and_crop(piece_mask, theta)
                piece_h, piece_w = rotated.shape
                
                # Random position, but ensure piece stays within bounds
                # Top-left corner can be from 0 to (width - piece_w) horizontally
                # and from 0 to (height - piece_h) vertically
                max_x = max(0, width - piece_w)
                max_y = max(0, height - piece_h)
                
                x = random.uniform(0, max_x)
                y = random.uniform(0, max_y)
                
                guess.append({
                    'piece_id': piece_id,
                    'x': x,
                    'y': y,
                    'theta': theta
                })
            
            # Sort by piece_id for consistency
            guess.sort(key=lambda x: x['piece_id'])
            guesses.append(guess)
        
        print(f"    ✓ Generated {len(guesses)} guesses")
        return guesses