from typing import Dict, List, Optional
import cv2
import numpy as np
from dataclasses import dataclass

from src.solver.corner_fitter import CornerFit, CornerFitter
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece


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


class IterativeSolver:
    """
    Iterative puzzle solver that tries corner pieces one at a time.
    Updates PuzzlePiece objects with final place_pose.
    
    ORIGINAL ALGORITHM - Now reads from enriched PuzzlePiece objects.
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
                          max_corner_combos: int = 1000) -> IterativeSolution:
        """
        Try different combinations of corners for each piece.
        Updates puzzle_pieces with place_pose when solution is found.
        
        NOW READS FROM: puzzle_pieces[i].corners (enriched data)
        INSTEAD OF: piece_corner_info dict
        """
        print("\n🔄 Starting iterative solving...")
        
        height, width = target.shape
        self.corner_fitter = CornerFitter(width=width, height=height)
        
        # Reset guess collection
        self.all_guesses = []
        self.all_scores = []
        
        # Find pieces with corners - READ FROM ENRICHED PUZZLEPIECE OBJECTS
        corner_pieces = [
            piece for piece in puzzle_pieces
            if piece.has_corner and len(piece.corners) > 0
        ]
        
        if not corner_pieces:
            print("  ⚠️  No corner pieces found!")
            return IterativeSolution(
                success=False,
                anchor_fit=None,
                remaining_placements=[],
                score=-float('inf'),
                iteration=0,
                total_iterations=0,
                all_guesses=[]
            )
        
        print(f"\n  Found {len(corner_pieces)} pieces with corners:")
        for piece in corner_pieces:
            print(f"    Piece {piece.id}: {len(piece.corners)} corners detected")
        
        # Generate all combinations
        import itertools
        piece_corner_options = []
        for piece in corner_pieces:
            piece_corner_options.append(list(range(len(piece.corners))))
        
        all_corner_combinations = list(itertools.product(*piece_corner_options))
        total_combos = len(all_corner_combinations)
        
        print(f"  Total corner combinations possible: {total_combos}")
        
        # Prioritize: try best corners first
        def combo_quality(combo_indices):
            total_quality = 0
            for piece, corner_idx in zip(corner_pieces, combo_indices):
                total_quality += piece.corners[corner_idx].quality
            return total_quality
        
        all_corner_combinations.sort(key=combo_quality, reverse=True)
        
        # Limit combinations
        if total_combos > max_corner_combos:
            print(f"  Limiting to best {max_corner_combos} combinations (by quality)")
            all_corner_combinations = all_corner_combinations[:max_corner_combos]
        
        best_score = -float('inf')
        best_guess = None
        no_improvement_count = 0
        last_best_score = -float('inf')
        combo_idx = 0
        
        # Adaptive parameters
        def get_adaptive_params(combo_idx, best_score, min_acceptable_score):
            if combo_idx < 100:
                num_guesses = 100 if best_score < min_acceptable_score else 50
                patience = 30
                check_interval = 50
            elif combo_idx < 400:
                if best_score < min_acceptable_score * 0.5:
                    num_guesses = 300
                elif best_score < min_acceptable_score:
                    num_guesses = 150
                else:
                    num_guesses = 75
                patience = 60
                check_interval = 30
            elif combo_idx < 4000:
                if best_score < min_acceptable_score * 0.5:
                    num_guesses = 400
                elif best_score < min_acceptable_score:
                    num_guesses = 200
                else:
                    num_guesses = 50
                patience = 100
                check_interval = 20
            else:
                if best_score < min_acceptable_score * 0.5:
                    num_guesses = 500
                elif best_score < min_acceptable_score:
                    num_guesses = 250
                else:
                    num_guesses = 50
                patience = 150
                check_interval = 10
            
            return num_guesses, patience, check_interval
        
        for combo_idx, corner_indices in enumerate(all_corner_combinations):
            num_guesses, patience, check_interval = get_adaptive_params(
                combo_idx, best_score, min_acceptable_score)
            
            print(f"\n  === Combination {combo_idx + 1}/{len(all_corner_combinations)} ===")
            print(f"  Adaptive params: {num_guesses} guesses, patience={patience}")
            
            combo_qual = combo_quality(corner_indices)
            print(f"  Corner indices: {corner_indices} (combined quality: {combo_qual:.3f})")
            
            # Build rotation mapping for this combination
            piece_rotations = {}
            
            for piece, corner_idx in zip(corner_pieces, corner_indices):
                selected_corner = piece.corners[corner_idx]
                selected_rotation = selected_corner.rotation_to_align
                
                print(f"    Piece {piece.id}: Corner #{corner_idx+1}, quality={selected_corner.quality:.3f}, rotation={selected_rotation:.1f}°")
                
                piece_rotations[int(piece.id)] = selected_rotation
            
            # Generate guesses
            guesses = self._generate_guesses_with_all_corners(
                piece_shapes,
                corner_pieces,
                piece_rotations,
                target,
                puzzle_pieces,
                max_guesses=num_guesses
            )
            
            if not guesses:
                continue
            
            # Score all guesses
            combo_best = -float('inf')
            for guess in guesses:
                self.all_guesses.append(guess)
                rendered = self.renderer.render(guess, piece_shapes)
                score = self.scorer.score(rendered, target)
                self.all_scores.append(score)
                
                if score > combo_best:
                    combo_best = score
                if score > best_score:
                    best_score = score
                    best_guess = guess
                    no_improvement_count = 0
            
            print(f"  Combo best: {combo_best:.1f}, Overall best: {best_score:.1f}")
            
            # Track improvement
            if best_score <= last_best_score:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            last_best_score = max(last_best_score, best_score)
            
            # Early exit
            if best_score >= score_threshold:
                print(f"\n  🎉 Found solution exceeding threshold ({score_threshold})!")
                break
            
            # Adaptive stopping
            should_stop_early = (
                combo_idx >= 50 and
                no_improvement_count >= patience and
                best_score >= min_acceptable_score
            )
            
            if should_stop_early:
                print(f"\n  ✓ Reached acceptable score ({best_score:.1f} >= {min_acceptable_score:.1f})")
                print(f"  No improvement for {no_improvement_count} combinations, stopping")
                break
            
            # Progress reports
            if combo_idx > 0 and combo_idx % check_interval == 0:
                if best_score < min_acceptable_score:
                    progress_pct = (best_score / min_acceptable_score) * 100
                    print(f"\n  📊 Progress at {combo_idx} combos:")
                    print(f"     Best score: {best_score:.1f} ({progress_pct:.1f}% of target)")
                    print(f"     No improvement streak: {no_improvement_count}")
                    print(f"     Total guesses tried: {len(self.all_guesses)}")
        
        total_combinations_tried = combo_idx + 1 if combo_idx >= 0 else 0
        
        print(f"\n🏆 Best solution: score {best_score:.1f}")
        print(f"📊 Tried {total_combinations_tried} corner combinations")
        print(f"📊 Total guesses: {len(self.all_guesses)}")
        
        # SUCCESS - Update PuzzlePiece objects with place_pose
        if best_guess:
            print(f"\n  ✓ Updating PuzzlePiece objects with place_pose...")
            piece_lookup = {int(p.id): p for p in puzzle_pieces}
            
            for placement in best_guess:
                piece_id = placement['piece_id']
                if piece_id in piece_lookup:
                    piece = piece_lookup[piece_id]
                    piece.place_pose = Pose(
                        x=placement['x'],
                        y=placement['y'],
                        theta=placement['theta']
                    )
                    print(f"    Piece {piece_id}: {piece.pick_pose} → {piece.place_pose}")
        
        success = best_score >= min_acceptable_score
        
        if not success:
            print(f"\n  ❌ Failed to reach minimum acceptable score of {min_acceptable_score:.1f}")
        elif best_score < score_threshold:
            print(f"\n  ⚠️  Reached acceptable score but not optimal threshold")
        
        return IterativeSolution(
            success=success,
            anchor_fit=None,
            remaining_placements=best_guess or [],
            score=best_score,
            iteration=total_combinations_tried,
            total_iterations=len(all_corner_combinations),
            all_guesses=self.all_guesses
        )
    
    def _generate_guesses_with_all_corners(self,
                                           piece_shapes: Dict[int, np.ndarray],
                                           corner_pieces: List[PuzzlePiece],
                                           piece_rotations: Dict[int, float],
                                           target: np.ndarray,
                                           all_pieces: List[PuzzlePiece],
                                           max_guesses: int = 10) -> List[List[dict]]:
        """Generate guesses where ALL pieces with corners are placed in corners."""
        import random
        
        height, width = target.shape
        
        if len(corner_pieces) == 0:
            return []
        
        if len(corner_pieces) > 4:
            corner_pieces = corner_pieces[:4]
        
        # Define the 4 corners
        corners = [
            ('bottom_right', width, height, 0),
            ('bottom_left', 0, height, 270),
            ('top_left', 0, 0, 180),
            ('top_right', width, 0, 90),
        ]
        
        # Get non-corner pieces
        corner_piece_ids = {int(p.id) for p in corner_pieces}
        non_corner_pieces = [
            p for p in all_pieces
            if int(p.id) not in corner_piece_ids
        ]
        
        guesses = []
        tried_permutations = set()
        
        for _ in range(max_guesses):
            shuffled_pieces = list(corner_pieces)
            random.shuffle(shuffled_pieces)
            
            perm_key = tuple(int(p.id) for p in shuffled_pieces)
            if perm_key in tried_permutations:
                continue
            tried_permutations.add(perm_key)
            
            guess = []
            
            # Place each corner piece
            available_corners = corners[:len(corner_pieces)]
            for piece, (corner_name, corner_x, corner_y, rotation_offset) in zip(shuffled_pieces, available_corners):
                piece_id = int(piece.id)
                rotation = piece_rotations[piece_id] + rotation_offset
                
                rotated_mask = self._rotate_and_crop(piece_shapes[piece_id], rotation)
                piece_h, piece_w = rotated_mask.shape
                
                # Calculate position
                if corner_name == 'top_left':
                    x, y = 0, 0
                elif corner_name == 'top_right':
                    x, y = width - piece_w, 0
                elif corner_name == 'bottom_left':
                    x, y = 0, height - piece_h
                else:  # bottom_right
                    x, y = width - piece_w, height - piece_h
                
                guess.append({
                    'piece_id': piece_id,
                    'x': float(x),
                    'y': float(y),
                    'theta': float(rotation)
                })
            
            # Place non-corner pieces randomly
            for piece in non_corner_pieces:
                piece_id = int(piece.id)
                theta = random.choice([0, 90, 180, 270])
                
                rotated = self._rotate_and_crop(piece_shapes[piece_id], theta)
                piece_h, piece_w = rotated.shape
                
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
            
            guess.sort(key=lambda p: p['piece_id'])
            guesses.append(guess)
        
        return guesses
    
    def _rotate_and_crop(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate and crop shape."""
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
        
        cropped = rotated[min_y:max_y+1, min_x:max_x+1]
        
        return cropped