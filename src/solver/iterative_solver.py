"""
Iterative solver with MODE SWITCHING
Switches between CORNER_SEARCH and EDGE_REFINEMENT modes adaptively.
"""

from typing import Dict, List, Optional
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.solver.corner_fitter import CornerFit, CornerFitter
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece


class SolverMode(Enum):
    """Solver operating modes."""
    CORNER_SEARCH = "corner_search"      # Rapidly evaluate corner layouts
    EDGE_REFINEMENT = "edge_refinement"  # Adaptive edge placement for best corners


@dataclass
class IterativeSolution:
    """Result from iterative solving process."""
    success: bool
    anchor_fit: Optional[CornerFit]
    remaining_placements: List[dict]
    score: float
    iteration: int
    total_iterations: int
    all_guesses: Optional[List[List[dict]]] = None


@dataclass
class SolverState:
    """Tracks solver state across mode switches."""
    mode: SolverMode
    corner_evaluations: List[tuple]  # (combo_idx, corner_indices, placements, score)
    best_score: float
    best_guess: Optional[List[dict]]
    iterations_since_improvement: int
    current_corner_rank: int  # Which corner layout we're currently refining
    refinement_attempts: int  # How many refinement attempts on current corner


class IterativeSolver:
    """
    Iterative puzzle solver with adaptive mode switching.
    
    MODES:
    - CORNER_SEARCH: Rapidly evaluate corner-only layouts in batches
    - EDGE_REFINEMENT: Adaptive edge placement on promising corners
    
    The solver switches between modes based on progress:
    - Starts in CORNER_SEARCH
    - Switches to EDGE_REFINEMENT when promising corners are found
    - Switches back to CORNER_SEARCH if refinement plateaus
    """

    def __init__(self, renderer, scorer, guess_generator):
        self.renderer = renderer
        self.scorer = scorer
        self.guess_generator = guess_generator
        self.corner_fitter = None
        self.all_guesses = []
        self.all_scores = []
    
    def solve_iteratively(self,
                          piece_shapes: Dict[int, np.ndarray],
                          target: np.ndarray,
                          puzzle_pieces: list,
                          score_threshold: float = 220000.0,
                          min_acceptable_score: float = 50000.0,
                          max_corner_combos: int = 1000,
                          # Mode switching parameters
                          mode_switching: bool = True,
                          initial_corner_count: int = 100,  # NEW: How many corners to evaluate upfront
                          max_corners_to_refine: int = 10,  # NEW: Max corners to try edge refinement on
                          refinement_patience: int = 5,
                          max_iterations: int = 500) -> IterativeSolution:
        """
        Solve puzzle with optional mode switching.
        
        NEW APPROACH:
        1. Evaluate many corners upfront (e.g., 100)
        2. Pick top N corners (e.g., top 10)  
        3. Do edge refinement on each
        4. Switch to next corner when stuck
        
        Args:
            mode_switching: Enable adaptive mode switching (recommended)
            initial_corner_count: How many corners to evaluate before refinement (default: 100)
            max_corners_to_refine: Max corners to try edge refinement on (default: 10)
            refinement_patience: Refinement attempts before moving to next corner (default: 5)
            max_iterations: Maximum total iterations across all modes
        """
        
        print("\n🔄 Starting iterative solving...")
        print(f"  Mode switching: {'ENABLED' if mode_switching else 'DISABLED'}")
        if mode_switching:
            print(f"  Config: {initial_corner_count} corners → refine top {max_corners_to_refine}, patience={refinement_patience}")
        
        height, width = target.shape
        self.corner_fitter = CornerFitter(width=width, height=height)
        
        # Reset state
        self.all_guesses = []
        self.all_scores = []
        
        # Find corner pieces - USE PIECE_TYPE CLASSIFICATION
        corner_pieces = [
            piece for piece in puzzle_pieces
            if piece.piece_type == "corner"
        ]
        
        if not corner_pieces:
            print("  ⚠️  No corner pieces found (piece_type == 'corner')!")
            # Fallback to has_corner if piece_type not set
            corner_pieces = [
                piece for piece in puzzle_pieces
                if piece.has_corner and len(piece.corners) > 0
            ]
            if not corner_pieces:
                return self._empty_solution()
            print(f"  ⚠️  Using fallback: found {len(corner_pieces)} pieces with corner features")
        
        print(f"\n  Found {len(corner_pieces)} corner pieces (by piece_type):")
        for piece in corner_pieces:
            print(f"    Piece {piece.id}: type={piece.piece_type}, {len(piece.corners)} corners")
        
        # Validate corner piece count
        if len(corner_pieces) != 4:
            print(f"\n  ⚠️  WARNING: Expected 4 corner pieces, found {len(corner_pieces)}!")
            if len(corner_pieces) < 4:
                print(f"      Not enough corner pieces - puzzle may not solve correctly")
            else:
                print(f"      Too many corner pieces - will only use first 4")
                corner_pieces = corner_pieces[:4]
        
        # Generate combinations:
        # 1. All permutations of which piece goes in which corner position
        # 2. For each permutation, try different corner rotations for each piece
        
        import itertools
        
        # Step 1: Permutations of pieces (which piece in which corner)
        piece_permutations = list(itertools.permutations(corner_pieces))
        
        print(f"\n  Piece permutations: {len(piece_permutations)} (which piece → which corner)")
        print(f"  Example: {[int(p.id) for p in piece_permutations[0]]}")
        print(f"           {[int(p.id) for p in piece_permutations[1]]}")
        
        # Step 2: For each permutation, generate corner rotation combinations
        all_corner_combinations = []
        
        for perm in piece_permutations:
            # For this permutation, get all rotation combinations
            piece_corner_options = [[i for i in range(len(p.corners))] for p in perm]
            rotation_combos = list(itertools.product(*piece_corner_options))
            
            # Store (piece_permutation, rotation_combo)
            for rotation_combo in rotation_combos:
                all_corner_combinations.append((perm, rotation_combo))
        
        print(f"\n  Total combinations: {len(all_corner_combinations)}")
        print(f"    = {len(piece_permutations)} permutations × avg rotations per perm")
        
        # Sort by quality (sum of corner qualities for each combo)
        def combo_quality(combo):
            perm, rotation_indices = combo
            total_quality = 0
            for piece, corner_idx in zip(perm, rotation_indices):
                total_quality += piece.corners[corner_idx].quality
            return total_quality
        
        all_corner_combinations.sort(key=combo_quality, reverse=True)
        
        if len(all_corner_combinations) > max_corner_combos:
            all_corner_combinations = all_corner_combinations[:max_corner_combos]
        
        print(f"  Total corner combinations: {len(all_corner_combinations)}")
        
        # Show piece distribution for verification
        all_edge_pieces = [p for p in puzzle_pieces if p.piece_type == "edge"]
        all_center_pieces = [p for p in puzzle_pieces if p.piece_type == "center"]
        
        print(f"\n  📋 Piece Distribution:")
        print(f"     Corners (→ placed in 4 corners): {[int(p.id) for p in corner_pieces]}")
        print(f"     Edges (→ placed along sides):   {[int(p.id) for p in all_edge_pieces]}")
        print(f"     Centers (→ placed in middle):   {[int(p.id) for p in all_center_pieces]}")
        
        # Solve with chosen mode
        if mode_switching:
            return self._solve_with_mode_switching(
                corner_pieces, all_corner_combinations, piece_shapes,
                target, puzzle_pieces, score_threshold, min_acceptable_score,
                initial_corner_count, max_corners_to_refine, 
                refinement_patience, max_iterations
            )
        else:
            return self._solve_classic(
                corner_pieces, all_corner_combinations, piece_shapes,
                target, puzzle_pieces, score_threshold, min_acceptable_score
            )
    
    def _solve_with_mode_switching(self,
                                   corner_pieces, all_corner_combinations,
                                   piece_shapes, target, puzzle_pieces,
                                   score_threshold, min_acceptable_score,
                                   initial_corner_count, max_corners_to_refine,
                                   refinement_patience, max_iterations) -> IterativeSolution:
        """Main mode-switching solve loop."""
        
        # ========================================================================
        # PHASE 1: EVALUATE MANY CORNERS FIRST (no edge refinement yet!)
        # ========================================================================
        print(f"\n  === PHASE 1: Evaluate corner layouts (no edges yet) ===")
        
        # Evaluate a large number of corners upfront
        initial_corners_to_evaluate = min(initial_corner_count, len(all_corner_combinations))
        print(f"  Will evaluate {initial_corners_to_evaluate} corner layouts before trying edges...")
        
        corner_evaluations = []
        
        for combo_idx in range(initial_corners_to_evaluate):
            piece_permutation, rotation_indices = all_corner_combinations[combo_idx]
            
            # Build rotations for this specific piece arrangement
            piece_rotations = {}
            for piece, corner_idx in zip(piece_permutation, rotation_indices):
                piece_rotations[int(piece.id)] = piece.corners[corner_idx].rotation_to_align
            
            # Place corners using this permutation
            corner_placements = self._place_corners(
                piece_permutation, piece_rotations, piece_shapes, target
            )
            
            # Score corner-only
            rendered = self.renderer.render(corner_placements, piece_shapes)
            score = self.scorer.score(rendered, target)
            
            # Store: (combo_idx, piece_perm, rotation_indices, placements, score)
            corner_evaluations.append((combo_idx, piece_permutation, rotation_indices, corner_placements, score))
            
            # ADD CORNER-ONLY PLACEMENT TO VISUALIZER
            self.all_guesses.append(corner_placements)
            self.all_scores.append(score)
            
            if (combo_idx + 1) % 25 == 0:
                # Show which pieces are where
                piece_ids = [int(p.id) for p in piece_permutation]
                print(f"    Evaluated {combo_idx + 1}/{initial_corners_to_evaluate} corners... (e.g. pieces {piece_ids})")
        
        # Sort by corner score
        corner_evaluations.sort(key=lambda x: x[4], reverse=True)  # x[4] is score now
        
        print(f"\n  📊 Top 10 corner layouts (corner-only scores):")
        for i, (idx, piece_perm, rotation_indices, _, score) in enumerate(corner_evaluations[:10]):
            piece_ids = [int(p.id) for p in piece_perm]
            print(f"    {i+1}. Combo {idx}: pieces {piece_ids}, score={score:.1f}")
        
        best_corner_score = corner_evaluations[0][4]  # score is at index 4
        worst_in_top10 = corner_evaluations[min(9, len(corner_evaluations)-1)][4]
        print(f"\n  Score range: {worst_in_top10:.1f} → {best_corner_score:.1f}")
        
        # ========================================================================
        # PHASE 2: SMART EDGE PLACEMENT
        # ========================================================================
        print(f"\n  === PHASE 2: Smart edge placement ===")
        
        # Get best corner layout
        _, best_piece_perm, best_rotation_indices, best_corner_placements, corner_only_score = corner_evaluations[0]
        
        print(f"  Starting with best corner layout: pieces {[int(p.id) for p in best_piece_perm]}")
        print(f"  Corner-only score: {corner_only_score:.1f}")
        
        # Get edge and center pieces
        corner_piece_ids = {int(p.id) for p in best_piece_perm}
        edge_pieces = [p for p in puzzle_pieces if p.piece_type == "edge" and int(p.id) not in corner_piece_ids]
        center_pieces = [p for p in puzzle_pieces if p.piece_type == "center" and int(p.id) not in corner_piece_ids]
        
        print(f"  Edge pieces to place: {[int(p.id) for p in edge_pieces]}")
        print(f"  Center pieces to place: {[int(p.id) for p in center_pieces]}")
        
        # Start with corner placements
        current_placements = best_corner_placements.copy()
        current_score = corner_only_score
        
        # Place each edge piece intelligently
        for edge_piece in edge_pieces:
            print(f"\n  → Placing edge piece {edge_piece.id}...")
            
            best_placement = self._find_best_edge_placement(
                edge_piece=edge_piece,
                piece_shapes=piece_shapes,
                current_placements=current_placements,
                target=target,
                current_score=current_score
            )
            
            if best_placement:
                current_placements.append(best_placement)
                
                # Score the new configuration
                rendered = self.renderer.render(current_placements, piece_shapes)
                new_score = self.scorer.score(rendered, target)
                improvement = new_score - current_score
                current_score = new_score
                
                # Add to visualizer
                self.all_guesses.append(current_placements.copy())
                self.all_scores.append(new_score)
                
                print(f"    ✓ Placed on {best_placement['side']} at ({best_placement['x']:.0f}, {best_placement['y']:.0f})")
                print(f"    Score: {new_score:.1f} ({improvement:+.1f})")
            else:
                print(f"    ⚠️  Could not find good placement for piece {edge_piece.id}")
        
        # Place center pieces (simple for now - just random)
        import random
        for center_piece in center_pieces:
            piece_id = int(center_piece.id)
            theta = 0
            rotated = self._rotate_and_crop(piece_shapes[piece_id], theta)
            piece_h, piece_w = rotated.shape
            
            x = random.uniform(50, target.shape[1] - piece_w - 50)
            y = random.uniform(50, target.shape[0] - piece_h - 50)
            
            current_placements.append({'piece_id': piece_id, 'x': x, 'y': y, 'theta': theta})
        
        # Final render and score
        rendered = self.renderer.render(current_placements, piece_shapes)
        final_score = self.scorer.score(rendered, target)
        
        # Add final to visualizer
        self.all_guesses.append(current_placements.copy())
        self.all_scores.append(final_score)
        
        print(f"\n🏆 Final score: {final_score:.1f}")
        print(f"   Improvement over corners: {final_score - corner_only_score:+.1f}")
        print(f"📊 Total guesses: {len(self.all_guesses)}")
        
        # Update piece poses
        self._update_piece_poses(puzzle_pieces, current_placements)
        
        return IterativeSolution(
            success=final_score >= min_acceptable_score,
            anchor_fit=None,
            remaining_placements=current_placements,
            score=final_score,
            iteration=initial_corners_to_evaluate + len(edge_pieces),
            total_iterations=len(all_corner_combinations),
            all_guesses=self.all_guesses
        )
    
    def _find_best_edge_placement(self,
                                  edge_piece: PuzzlePiece,
                                  piece_shapes: Dict[int, np.ndarray],
                                  current_placements: List[dict],
                                  target: np.ndarray,
                                  current_score: float) -> Optional[dict]:
        """
        Smart edge placement:
        1. Try each side (right, left, top, bottom) with all 4 rotations
        2. If side doesn't improve score → abort that side immediately
        3. If side improves → slide along axis to find best position
        """
        
        piece_id = int(edge_piece.id)
        height, width = target.shape
        
        # Define sides to try - try all 4 rotations per side
        sides = [
            'right',
            'left',
            'top',
            'bottom',
        ]
        
        rotations = [0, 90, 180, 270]
        
        best_placement = None
        best_score = current_score
        
        for side_name in sides:
            print(f"    Trying {side_name} side...")
            
            side_best_score = current_score
            side_best_placement = None
            
            # Try all rotations for this side
            for rotation in rotations:
                rotated_mask = self._rotate_and_crop(piece_shapes[piece_id], rotation)
                piece_h, piece_w = rotated_mask.shape
                
                # Initial test position for this side
                if side_name == 'right':
                    test_x = width - piece_w
                    test_y = (height - piece_h) / 2
                    axis_type = 'vertical'
                elif side_name == 'left':
                    test_x = 0
                    test_y = (height - piece_h) / 2
                    axis_type = 'vertical'
                elif side_name == 'bottom':
                    test_x = (width - piece_w) / 2
                    test_y = height - piece_h
                    axis_type = 'horizontal'
                else:  # top
                    test_x = (width - piece_w) / 2
                    test_y = 0
                    axis_type = 'horizontal'
                
                # Test initial position
                test_placement = {
                    'piece_id': piece_id,
                    'x': test_x,
                    'y': test_y,
                    'theta': rotation,
                    'side': side_name
                }
                
                test_placements = current_placements + [test_placement]
                
                # ADD TEST TO VISUALIZER
                self.all_guesses.append(test_placements.copy())
                
                rendered = self.renderer.render(test_placements, piece_shapes)
                test_score = self.scorer.score(rendered, target)
                self.all_scores.append(test_score)
                
                if test_score > side_best_score:
                    side_best_score = test_score
                    side_best_placement = test_placement.copy()
                    side_best_placement['axis_type'] = axis_type
            
            print(f"      Best rotation: θ={side_best_placement['theta'] if side_best_placement else 'N/A'}°, score={side_best_score:.1f}")
            
            # If no improvement on this side, skip sliding
            if side_best_score <= current_score:
                print(f"      → No improvement, skip {side_name}")
                continue
            
            # This side improved! Slide along axis to optimize
            print(f"      → {side_name} improved! Sliding to optimize...")
            print(f"      → {side_name} improved! Sliding to optimize...")
            
            optimized = self._slide_along_axis(
                piece_id=piece_id,
                piece_shapes=piece_shapes,
                current_placements=current_placements,
                target=target,
                initial_placement=side_best_placement,
                axis_type=side_best_placement['axis_type'],
                side_name=side_name
            )
            
            if optimized['score'] > best_score:
                best_score = optimized['score']
                best_placement = optimized['placement']
                print(f"      → Optimized score: {best_score:.1f}")
        
        return best_placement
    
    def _slide_along_axis(self,
                         piece_id: int,
                         piece_shapes: Dict[int, np.ndarray],
                         current_placements: List[dict],
                         target: np.ndarray,
                         initial_placement: dict,
                         axis_type: str,
                         side_name: str) -> dict:
        """
        Slide piece along its axis (vertical or horizontal) to find best position.
        Uses a simple grid search with 20 test positions.
        
        IMPORTANT: Adds each test position to visualizer!
        """
        
        height, width = target.shape
        rotated_mask = self._rotate_and_crop(piece_shapes[piece_id], initial_placement['theta'])
        piece_h, piece_w = rotated_mask.shape
        
        best_placement = initial_placement.copy()
        best_score = -float('inf')
        
        # Determine search range
        if axis_type == 'vertical':
            # Slide up/down (vary y)
            positions = np.linspace(0, max(0, height - piece_h), num=20)
            
            for y_pos in positions:
                test_placement = initial_placement.copy()
                test_placement['y'] = float(y_pos)
                
                test_placements = current_placements + [test_placement]
                rendered = self.renderer.render(test_placements, piece_shapes)
                score = self.scorer.score(rendered, target)
                
                # ADD TO VISUALIZER
                self.all_guesses.append(test_placements.copy())
                self.all_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_placement = test_placement.copy()
        
        else:  # horizontal
            # Slide left/right (vary x)
            positions = np.linspace(0, max(0, width - piece_w), num=20)
            
            for x_pos in positions:
                test_placement = initial_placement.copy()
                test_placement['x'] = float(x_pos)
                
                test_placements = current_placements + [test_placement]
                rendered = self.renderer.render(test_placements, piece_shapes)
                score = self.scorer.score(rendered, target)
                
                # ADD TO VISUALIZER
                self.all_guesses.append(test_placements.copy())
                self.all_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_placement = test_placement.copy()
        
        return {'placement': best_placement, 'score': best_score}
    
    def _solve_classic(self, corner_pieces, all_corner_combinations,
                      piece_shapes, target, puzzle_pieces,
                      score_threshold, min_acceptable_score) -> IterativeSolution:
        """Classic approach: evaluate all corners, return best (no edges for now)."""
        
        print("\n  === Evaluating all corners (classic mode - corners only) ===")
        
        corner_evaluations = []
        
        for combo_idx, combo in enumerate(all_corner_combinations):
            piece_permutation, rotation_indices = combo
            
            # Build rotations
            piece_rotations = {}
            for piece, corner_idx in zip(piece_permutation, rotation_indices):
                piece_rotations[int(piece.id)] = piece.corners[corner_idx].rotation_to_align
            
            # Place corners
            corner_placements = self._place_corners(
                piece_permutation, piece_rotations, piece_shapes, target
            )
            
            # Score
            rendered = self.renderer.render(corner_placements, piece_shapes)
            score = self.scorer.score(rendered, target)
            
            corner_evaluations.append((combo_idx, piece_permutation, rotation_indices, corner_placements, score))
            
            # Add to visualizer
            self.all_guesses.append(corner_placements)
            self.all_scores.append(score)
            
            if (combo_idx + 1) % 50 == 0:
                print(f"    Evaluated {combo_idx + 1} corners...")
        
        corner_evaluations.sort(key=lambda x: x[4], reverse=True)
        
        print(f"\n  Top 5 corners:")
        for i, (idx, piece_perm, _, _, score) in enumerate(corner_evaluations[:5]):
            piece_ids = [int(p.id) for p in piece_perm]
            print(f"    {i+1}. Combo {idx}: pieces {piece_ids}, score={score:.1f}")
        
        # Return best corner (no edges)
        _, best_piece_perm, _, best_placements, best_score = corner_evaluations[0]
        
        self._update_piece_poses(puzzle_pieces, best_placements)
        
        return IterativeSolution(
            success=best_score >= min_acceptable_score,
            anchor_fit=None,
            remaining_placements=best_placements,
            score=best_score,
            iteration=len(corner_evaluations),
            total_iterations=len(all_corner_combinations),
            all_guesses=self.all_guesses
        )
    
    def _place_corners(self, corner_pieces, piece_rotations, piece_shapes, target):
        """Place corner pieces in the 4 corners."""
        
        height, width = target.shape
        
        corners = [
            ('bottom_right', width, height, 0),
            ('bottom_left', 0, height, 270),
            ('top_left', 0, 0, 180),
            ('top_right', width, 0, 90),
        ]
        
        placements = []
        
        for piece, (corner_name, corner_x, corner_y, rotation_offset) in \
                zip(corner_pieces, corners[:len(corner_pieces)]):
            
            piece_id = int(piece.id)
            rotation = piece_rotations[piece_id] + rotation_offset
            
            rotated = self._rotate_and_crop(piece_shapes[piece_id], rotation)
            piece_h, piece_w = rotated.shape
            
            if corner_name == 'top_left':
                x, y = 0, 0
            elif corner_name == 'top_right':
                x, y = width - piece_w, 0
            elif corner_name == 'bottom_left':
                x, y = 0, height - piece_h
            else:  # bottom_right
                x, y = width - piece_w, height - piece_h
            
            placements.append({
                'piece_id': piece_id,
                'x': float(x),
                'y': float(y),
                'theta': float(rotation)
            })
        
        return placements
    
    def _try_edge_placements(self, corner_placements, piece_shapes,
                            target, puzzle_pieces, corner_piece_ids,
                            num_attempts):
        """Try multiple edge placement strategies."""
        
        best_score = -float('inf')
        best_placement = None
        
        for attempt in range(num_attempts):
            guesses = self._generate_guesses_for_corner_layout(
                piece_shapes, corner_placements, target,
                puzzle_pieces, corner_piece_ids, num_variations=5
            )
            
            for guess in guesses:
                self.all_guesses.append(guess)
                rendered = self.renderer.render(guess, piece_shapes)
                score = self.scorer.score(rendered, target)
                self.all_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_placement = guess
        
        return {'score': best_score, 'placement': best_placement}
    
    def _generate_guesses_for_corner_layout(self, piece_shapes, corner_placements,
                                           target, all_pieces, corner_piece_ids,
                                           num_variations):
        """Generate guesses for a fixed corner layout by placing edges/centers."""
        
        import random
        
        height, width = target.shape
        
        # Separate pieces by type
        edge_pieces = [p for p in all_pieces 
                      if int(p.id) not in corner_piece_ids and p.piece_type == "edge"]
        center_pieces = [p for p in all_pieces
                        if int(p.id) not in corner_piece_ids and p.piece_type == "center"]
        
        print(f"      Generating guesses: {len(edge_pieces)} edges, {len(center_pieces)} centers")
        
        guesses = []
        seen_signatures = set()
        attempts = 0
        max_attempts = num_variations * 10  # Try harder to find unique placements
        
        while len(guesses) < num_variations and attempts < max_attempts:
            attempts += 1
            guess = corner_placements.copy()
            
            # Place edges with varied strategies
            edge_strategy = attempts % 4  # Rotate through 4 strategies
            
            for edge_idx, piece in enumerate(edge_pieces):
                piece_id = int(piece.id)
                
                # Use piece's primary edge rotation if available
                if piece.primary_edge_rotation is not None:
                    base_rotation = piece.primary_edge_rotation
                    theta = base_rotation + (edge_strategy * 90)  # Vary by strategy
                else:
                    theta = random.choice([0, 90, 180, 270])
                
                rotated = self._rotate_and_crop(piece_shapes[piece_id], theta)
                piece_h, piece_w = rotated.shape
                
                # Use different placement strategies
                if edge_strategy == 0:
                    # Strategy 1: Edges along sides
                    if edge_idx == 0:
                        x = 0  # Left
                        y = random.uniform(0, max(0, height - piece_h))
                    else:
                        x = max(0, width - piece_w)  # Right
                        y = random.uniform(0, max(0, height - piece_h))
                elif edge_strategy == 1:
                    # Strategy 2: Edges top/bottom
                    if edge_idx == 0:
                        x = random.uniform(0, max(0, width - piece_w))
                        y = 0  # Top
                    else:
                        x = random.uniform(0, max(0, width - piece_w))
                        y = max(0, height - piece_h)  # Bottom
                elif edge_strategy == 2:
                    # Strategy 3: Diagonal positions
                    x = random.uniform(0, max(0, width - piece_w))
                    y = random.uniform(0, max(0, height - piece_h))
                else:
                    # Strategy 4: Center-biased
                    center_x = (width - piece_w) / 2
                    center_y = (height - piece_h) / 2
                    x = center_x + random.uniform(-100, 100)
                    y = center_y + random.uniform(-100, 100)
                    x = max(0, min(width - piece_w, x))
                    y = max(0, min(height - piece_h, y))
                
                guess.append({'piece_id': piece_id, 'x': x, 'y': y, 'theta': theta})
            
            # Place centers with variation
            for center_idx, piece in enumerate(center_pieces):
                piece_id = int(piece.id)
                theta = random.choice([0, 90, 180, 270])
                
                rotated = self._rotate_and_crop(piece_shapes[piece_id], theta)
                piece_h, piece_w = rotated.shape
                
                # Add noise to position based on attempt number
                noise = (attempts % 10) * 10  # Vary noise by attempt
                
                x = random.uniform(0, max(0, width - piece_w)) + random.uniform(-noise, noise)
                y = random.uniform(0, max(0, height - piece_h)) + random.uniform(-noise, noise)
                
                x = max(0, min(width - piece_w, x))
                y = max(0, min(height - piece_h, y))
                
                guess.append({'piece_id': piece_id, 'x': x, 'y': y, 'theta': theta})
            
            guess.sort(key=lambda p: p['piece_id'])
            
            # Create signature for deduplication (round to nearest 10 pixels)
            signature = tuple(sorted((p['piece_id'], 
                                     round(p['x'] / 10), 
                                     round(p['y'] / 10), 
                                     p['theta']) for p in guess))
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                guesses.append(guess)
        
        if len(guesses) < num_variations:
            print(f"      Warning: Only generated {len(guesses)}/{num_variations} unique guesses")
        
        return guesses
    
    def _update_piece_poses(self, puzzle_pieces, placements):
        """Update PuzzlePiece objects with place_pose."""
        
        piece_lookup = {int(p.id): p for p in puzzle_pieces}
        
        for placement in placements:
            piece_id = placement['piece_id']
            if piece_id in piece_lookup:
                piece_lookup[piece_id].place_pose = Pose(
                    x=placement['x'],
                    y=placement['y'],
                    theta=placement['theta']
                )
    
    def _rotate_and_crop(self, shape, angle):
        """Rotate and crop piece shape."""
        
        if angle == 0:
            return shape
        
        h, w = shape.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(shape, M, (new_w, new_h))
        
        # Crop to content
        piece_points = np.argwhere(rotated > 0)
        if len(piece_points) == 0:
            return rotated
        
        min_y, min_x = piece_points.min(axis=0)
        max_y, max_x = piece_points.max(axis=0)
        
        return rotated[min_y:max_y+1, min_x:max_x+1]
    
    def _empty_solution(self):
        """Return empty solution."""
        return IterativeSolution(
            success=False,
            anchor_fit=None,
            remaining_placements=[],
            score=-float('inf'),
            iteration=0,
            total_iterations=0,
            all_guesses=[]
        )