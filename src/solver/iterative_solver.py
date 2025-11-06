# src/solver/iterative_solver.py

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

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
        self.corner_fitter = CornerFitter(width=800, height=800)
    
    def solve_iteratively(self,
                         piece_shapes: Dict[int, np.ndarray],
                         piece_corner_info: Dict[int, PieceCornerInfo],
                         target: np.ndarray,
                         score_threshold: float = 50000.0,
                         max_iterations: int = 6) -> IterativeSolution:
        """
        Try pieces iteratively until we find a good solution.
        
        Args:
            piece_shapes: Dict of piece masks
            piece_corner_info: Corner analysis results
            target: Target layout
            score_threshold: Stop if we achieve this score
            max_iterations: Maximum pieces to try (None = try all)
            
        Returns:
            Best solution found
        """
        print("\n🔄 Starting iterative solving...")
        
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
            
            if solution.success:
                print(f"✓ Found valid solution with score {solution.score:.1f}")
                
                # Keep best solution
                if best_solution is None or solution.score > best_solution.score:
                    best_solution = solution
                
                # If score is good enough, we can stop
                if solution.score >= score_threshold:
                    print(f"🎉 Score exceeds threshold ({score_threshold}), stopping!")
                    break
            else:
                print(f"✗ Solution failed with score {solution.score:.1f}")
        
        if best_solution is None:
            # Return failed solution
            return IterativeSolution(
                success=False,
                anchor_fit=None,
                remaining_placements=[],
                score=-float('inf'),
                iteration=len(candidates),
                total_iterations=len(candidates)
            )
        
        print(f"\n🏆 Best solution: score {best_solution.score:.1f}")
        return best_solution
    # Update _try_piece_as_anchor in iterative_solver.py

    def _try_piece_as_anchor(self,
                            candidate: PieceCornerInfo,
                            piece_shapes: Dict[int, np.ndarray],
                            target: np.ndarray) -> IterativeSolution:
        """
        Try a specific piece as the anchor corner piece.
        Use the bottom-right corner and the pre-calculated rotation.
        
        Returns:
            Solution with this piece anchored
        """
        piece_id = candidate.piece_id
        piece_mask = piece_shapes[piece_id]
        
        # Get bottom-right corner of target
        corners = self.corner_fitter.identify_target_corners(target)
        # corners[3] is bottom-right (top-left, top-right, bottom-left, bottom-right)
        corner_x, corner_y, corner_type = corners[3]
        
        print(f"  Fitting to {corner_type} corner at ({corner_x:.0f}, {corner_y:.0f})")
        
        # Use the pre-calculated rotation if available
        if candidate.has_corner and candidate.rotation_to_bottom_right is not None:
            initial_rotation = candidate.rotation_to_bottom_right
            print(f"  Using pre-calculated rotation: {initial_rotation:.1f}°")
            
            # Fine-tune around the pre-calculated rotation
            best_fit = self._fine_tune_rotation(
                piece_id,
                piece_mask,
                (corner_x, corner_y),
                initial_rotation,
                target
            )
        else:
            # No corner detected, do full search
            print(f"  No corner detected, doing full rotation search")
            best_fit = self.corner_fitter.fit_piece_to_corner(
                piece_id,
                piece_mask,
                (corner_x, corner_y),
                corner_type,
                target
            )
        
        print(f"  Corner fit score: {best_fit.score:.1f}, final rotation: {best_fit.rotation:.1f}°")
        
        if best_fit.score < -50000:
            # This piece doesn't fit well as a corner
            return IterativeSolution(
                success=False,
                anchor_fit=best_fit,
                remaining_placements=[],
                score=best_fit.score,
                iteration=0,
                total_iterations=1
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
        
        # Find best placement for remaining pieces
        best_remaining_score = -float('inf')
        best_guess = None
        
        for i, guess in enumerate(guesses):
            rendered = self.renderer.render(guess, piece_shapes)
            score = self.scorer.score(rendered, target)
            
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
                total_iterations=1
            )
        else:
            return IterativeSolution(
                success=False,
                anchor_fit=best_fit,
                remaining_placements=[],
                score=-float('inf'),
                iteration=0,
                total_iterations=1
            )


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
        
        return CornerFit(
            piece_id=piece_id,
            corner_position=corner_pos,
            rotation=float(best_rotation),  # Cast to float
            score=float(best_score)  # Cast to float
        )

    def _generate_guesses_with_anchor(self,
                                     anchor_fit: CornerFit,
                                     remaining_piece_ids: List[int],
                                     piece_shapes: Dict[int, np.ndarray],
                                     target: np.ndarray,
                                     max_guesses: int = 5000) -> List[List[dict]]:
        """
        Generate guesses with one piece fixed as anchor.
        """
        from itertools import permutations, product
        import random
        
        # Generate positions for remaining pieces (focused on target)
        positions = self.guess_generator.generate_grid_positions(target, grid_spacing=50)
        
        # Sample positions to limit combinatorial explosion
        sampled_positions = random.sample(positions, min(len(positions), 10))
        
        # Rotations
        rotations = list(range(0, 360, 15))
        
        guesses = []
        
        # Limit permutations
        piece_perms = list(permutations(remaining_piece_ids))
        if len(piece_perms) > 6:
            random.shuffle(piece_perms)
            piece_perms = piece_perms[:6]
        
        for piece_order in piece_perms:
            for rotation_combo in product(rotations, repeat=len(remaining_piece_ids)):
                for position_combo in product(sampled_positions, repeat=len(remaining_piece_ids)):
                    guess = []
                    
                    # Add ANCHORED piece first
                    guess.append({
                        'piece_id': anchor_fit.piece_id,
                        'x': anchor_fit.corner_position[0],
                        'y': anchor_fit.corner_position[1],
                        'theta': anchor_fit.rotation
                    })
                    
                    # Add remaining pieces
                    for piece_id, pos, theta in zip(piece_order, position_combo, rotation_combo):
                        guess.append({
                            'piece_id': piece_id,
                            'x': pos[0],
                            'y': pos[1],
                            'theta': theta
                        })
                    
                    # Sort by piece_id for consistency
                    guess.sort(key=lambda x: x['piece_id'])
                    guesses.append(guess)
                    
                    if len(guesses) >= max_guesses:
                        return guesses
        
        return guesses