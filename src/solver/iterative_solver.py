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
                     score_threshold: float = 230000.0,
                     min_acceptable_score: float = 200000.0,
                     max_corner_combos: int = 200) -> IterativeSolution:
        """
        Try different combinations of corners for each piece.
        """
        print("\n🔄 Starting iterative solving...")
        
        height, width = target.shape
        self.corner_fitter = CornerFitter(width=width, height=height)
        
        # Reset guess collection
        self.all_guesses = []
        self.all_scores = []
        
        # Find pieces with corners
        corner_pieces = [
            (piece_id, info) 
            for piece_id, info in piece_corner_info.items() 
            if info.has_corner and len(info.corner_rotations) > 0
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
        for piece_id, info in corner_pieces:
            print(f"    Piece {piece_id}: {len(info.corner_rotations)} corners detected")
        
        # Generate all combinations of corner selections
        import itertools
        piece_corner_options = []
        for piece_id, info in corner_pieces:
            piece_corner_options.append(list(range(len(info.corner_rotations))))
        
        all_corner_combinations = list(itertools.product(*piece_corner_options))
        total_combos = len(all_corner_combinations)
        
        print(f"  Total corner combinations possible: {total_combos}")
        
        # Prioritize: try best corners first
        def combo_quality(combo_indices):
            total_quality = 0
            for (piece_id, info), corner_idx in zip(corner_pieces, combo_indices):
                total_quality += info.corner_qualities[corner_idx].overall_score
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
        
        for combo_idx, corner_indices in enumerate(all_corner_combinations):
            print(f"\n  === Combination {combo_idx + 1}/{len(all_corner_combinations)} ===")
            
            # Calculate quality of this combination
            combo_qual = combo_quality(corner_indices)
            print(f"  Corner indices: {corner_indices} (combined quality: {combo_qual:.3f})")
            
            # Create modified corner info using selected corners
            modified_corner_info = {}
            
            for (piece_id, info), corner_idx in zip(corner_pieces, corner_indices):
                selected_rotation = info.corner_rotations[corner_idx]
                selected_quality = info.corner_qualities[corner_idx]
                
                print(f"    Piece {piece_id}: Corner #{corner_idx+1}, quality={selected_quality.overall_score:.3f}, rotation={selected_rotation:.1f}°")
                
                modified_corner_info[piece_id] = PieceCornerInfo(
                    piece_id=piece_id,
                    has_corner=True,
                    corner_count=1,
                    corner_positions=[selected_quality.position],
                    corner_qualities=[selected_quality],
                    corner_rotations=[selected_rotation],
                    primary_corner_angle=None,
                    rotation_to_bottom_right=selected_rotation,
                    piece_center=info.piece_center
                )
            
            # Generate MORE guesses per combo - more chances to get non-corner pieces right
            # Increase attempts dramatically if we haven't reached minimum acceptable score
            if best_score < min_acceptable_score:
                num_guesses = 100 if combo_idx < 20 else 50
            else:
                num_guesses = 50 if combo_idx < 10 else 20
            
            guesses = self._generate_guesses_with_all_corners(
                piece_shapes,
                modified_corner_info,
                target,
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
                    no_improvement_count = 0  # Reset counter
            
            print(f"  Combo best: {combo_best:.1f}, Overall best: {best_score:.1f}")
            
            # Track improvement
            if best_score <= last_best_score:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # Reset when we improve
            last_best_score = max(last_best_score, best_score)
            
            # Early exit if we found a great solution
            if best_score >= score_threshold:
                print(f"\n  🎉 Found solution exceeding threshold ({score_threshold})!")
                break
            
            # MORE LENIENT stopping conditions
            # Only stop early if:
            # 1. We've tried at least 50 combos
            # 2. No improvement for 30 consecutive combos
            # 3. AND we've reached at least the minimum acceptable score
            should_stop_early = (
                combo_idx >= 50 and
                no_improvement_count >= 30 and
                best_score >= min_acceptable_score
            )
            
            if should_stop_early:
                print(f"\n  ✓ Reached acceptable score ({best_score:.1f} >= {min_acceptable_score:.1f})")
                print(f"  No improvement for {no_improvement_count} combinations, stopping")
                break
            
            # Warn if we're stuck below minimum acceptable score
            if combo_idx > 0 and combo_idx % 25 == 0 and best_score < min_acceptable_score:
                print(f"\n  ⚠️  After {combo_idx} combos, best score is {best_score:.1f} (target: {min_acceptable_score:.1f})")
                print(f"  Continuing search...")
        
        # Calculate final values
        total_combinations_tried = combo_idx + 1 if combo_idx >= 0 else 0
        
        print(f"\n🏆 Best solution: score {best_score:.1f}")
        print(f"📊 Tried {total_combinations_tried} corner combinations")
        print(f"📊 Total guesses: {len(self.all_guesses)}")
        
        # Success criteria
        success = best_score >= min_acceptable_score
        
        if not success:
            print(f"\n  ❌ Failed to reach minimum acceptable score of {min_acceptable_score:.1f}")
            if best_score > 50000:
                print(f"  💡 Score {best_score:.1f} suggests corner pieces might be correct")
                print(f"     but edge/center pieces are misplaced.")
                print(f"     The solver tried {total_combinations_tried} corner combinations.")
        elif best_score < score_threshold:
            print(f"\n  ⚠️  Reached acceptable score but not optimal threshold")
            print(f"     Acceptable: {min_acceptable_score:.1f}, Achieved: {best_score:.1f}, Target: {score_threshold:.1f}")
        
        return IterativeSolution(
            success=success,
            anchor_fit=None,
            remaining_placements=best_guess or [],
            score=best_score,
            iteration=total_combinations_tried,
            total_iterations=len(all_corner_combinations),
            all_guesses=self.all_guesses
        )
    """"
    def _try_piece_as_anchor(self,
                        candidate: PieceCornerInfo,
                        piece_shapes: Dict[int, np.ndarray],
                        target: np.ndarray) -> IterativeSolution:

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
"""
    def _generate_guesses_with_all_corners(self,
                                        piece_shapes: Dict[int, np.ndarray],
                                        piece_corner_info: Dict[int, PieceCornerInfo],
                                        target: np.ndarray,
                                        max_guesses: int = 10) -> List[List[dict]]:
        """
        Generate guesses where ALL pieces with corners are placed in corners.
        Each corner piece goes to a different corner.
        """
        import random
        
        height, width = target.shape
        
        print(f"\n  === GENERATING GUESSES WITH ALL CORNER PIECES ===")
        print(f"  Target size: {width}x{height}")
        
        # Find all pieces with corners
        corner_pieces = [
            (piece_id, info) 
            for piece_id, info in piece_corner_info.items() 
            if info.has_corner and info.rotation_to_bottom_right is not None
        ]
        
        if len(corner_pieces) == 0:
            print("  ⚠️  No corner pieces found!")
            return []
        
        if len(corner_pieces) > 4:
            print(f"  ⚠️  Found {len(corner_pieces)} corner pieces, but only 4 corners available!")
            corner_pieces = corner_pieces[:4]
        
        print(f"  Found {len(corner_pieces)} corner pieces: {[p[0] for p in corner_pieces]}")
        
        # Define the 4 corners with their rotation offsets from bottom-right
        corners = [
            ('bottom_right', width, height, 0),      # rotation_to_bottom_right + 0
            ('bottom_left', 0, height, 270),          # rotation_to_bottom_right + 90
            ('top_left', 0, 0, 180),                 # rotation_to_bottom_right + 180
            ('top_right', width, 0, 90),            # rotation_to_bottom_right + 270
        ]
        
        # Get non-corner pieces
        corner_piece_ids = {p[0] for p in corner_pieces}
        non_corner_piece_ids = [
            pid for pid in piece_shapes.keys() 
            if pid not in corner_piece_ids
        ]
        
        print(f"  Non-corner pieces: {non_corner_piece_ids}")
        
        guesses = []
        num_permutations = max_guesses
        
        if len(corner_pieces) <= len(corners):
            available_corners = corners[:len(corner_pieces)]
            tried_permutations = set()
            
            for _ in range(num_permutations):
                shuffled_pieces = list(corner_pieces)
                random.shuffle(shuffled_pieces)
                
                perm_key = tuple(p[0] for p in shuffled_pieces)
                if perm_key in tried_permutations:
                    continue
                tried_permutations.add(perm_key)
                
                guess = []
                
                # Place each corner piece in a corner
                for (piece_id, piece_info), (corner_name, corner_x, corner_y, rotation_offset) in zip(shuffled_pieces, available_corners):
                    # rotation_to_bottom_right tells us how to rotate the ORIGINAL piece to fit bottom-right
                    # To fit a different corner, just add the offset
                    rotation = piece_info.rotation_to_bottom_right + rotation_offset
                    
                    print(f"    Piece {piece_id} -> {corner_name}:")
                    print(f"      rotation_to_bottom_right: {piece_info.rotation_to_bottom_right:.1f}°")
                    print(f"      Additional offset: {rotation_offset}°")
                    print(f"      Total rotation: {rotation:.1f}°")
                    
                    # Rotate from original piece with total rotation
                    rotated_mask = self._rotate_and_crop(piece_shapes[piece_id], rotation)
                    piece_h, piece_w = rotated_mask.shape
                    
                    print(f"      Piece size after rotation: {piece_w}x{piece_h}")
                    
                    # Calculate position based on corner
                    if corner_name == 'top_left':
                        x, y = 0, 0
                    elif corner_name == 'top_right':
                        x, y = width - piece_w, 0
                    elif corner_name == 'bottom_left':
                        x, y = 0, height - piece_h
                    else:  # bottom_right
                        x, y = width - piece_w, height - piece_h
                    
                    print(f"      Position: ({x}, {y})")
                    
                    guess.append({
                        'piece_id': piece_id,
                        'x': float(x),
                        'y': float(y),
                        'theta': float(rotation)
                    })
                
                # Place remaining non-corner pieces randomly
                for piece_id in non_corner_piece_ids:
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
        
        print(f"  ✓ Generated {len(guesses)} guesses with corner pieces in corners")
        return guesses
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