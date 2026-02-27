"""
Edge placement logic for the iterative solver.
Handles edge piece positioning, sliding optimization, and center piece placement.
"""

import random
from typing import Dict, List, Optional

import numpy as np

from src.utils.geometry import rotate_and_crop
from src.utils.puzzle_piece import PuzzlePiece


def try_edge_placement_on_corners(
    corner_pieces,
    corner_placements,
    corner_only_score,
    piece_shapes,
    target,
    puzzle_pieces,
    layout_number,
    renderer,
    scorer,
    all_guesses,
    all_scores,
) -> dict:
    """Try smart edge placement on a specific corner layout."""

    # Get edge and center pieces
    corner_piece_ids = {int(p.id) for p in corner_pieces}
    edge_pieces = [
        p
        for p in puzzle_pieces
        if p.piece_type == "edge" and int(p.id) not in corner_piece_ids
    ]
    center_pieces = [
        p
        for p in puzzle_pieces
        if p.piece_type == "center" and int(p.id) not in corner_piece_ids
    ]

    print(f"    Edge pieces to place: {[int(p.id) for p in edge_pieces]}")

    # Start with corner placements
    current_placements = corner_placements.copy()
    current_score = corner_only_score

    # Place each edge piece intelligently
    for edge_piece in edge_pieces:
        print(f"      → Placing edge piece {edge_piece.id}...")

        best_placement = find_best_edge_placement(
            edge_piece=edge_piece,
            piece_shapes=piece_shapes,
            current_placements=current_placements,
            target=target,
            current_score=current_score,
            renderer=renderer,
            scorer=scorer,
            all_guesses=all_guesses,
            all_scores=all_scores,
        )

        if best_placement:
            current_placements.append(best_placement)

            # Score the new configuration
            rendered = renderer.render(current_placements, piece_shapes)
            new_score = scorer.score(rendered, target)
            improvement = new_score - current_score
            current_score = new_score

            # Add to visualizer
            all_guesses.append(current_placements.copy())
            all_scores.append(new_score)

            print(
                f"        ✓ Placed on {best_placement['side']} at ({best_placement['x']:.0f}, {best_placement['y']:.0f})"
            )
            print(f"        Score: {new_score:.1f} ({improvement:+.1f})")
        else:
            print(
                f"        ⚠️  Could not find good placement for piece {edge_piece.id}"
            )

    # Place center pieces (simple for now - just random)
    for center_piece in center_pieces:
        piece_id = int(center_piece.id)
        theta = 0
        rotated = rotate_and_crop(piece_shapes[piece_id], theta)
        piece_h, piece_w = rotated.shape

        x = random.uniform(50, target.shape[1] - piece_w - 50)
        y = random.uniform(50, target.shape[0] - piece_h - 50)

        current_placements.append(
            {"piece_id": piece_id, "x": x, "y": y, "theta": theta}
        )

    # Final render and score
    rendered = renderer.render(current_placements, piece_shapes)
    final_score = scorer.score(rendered, target)

    # Add final to visualizer
    all_guesses.append(current_placements.copy())
    all_scores.append(final_score)

    return {
        "final_score": final_score,
        "final_placements": current_placements,
        "improvement": final_score - corner_only_score,
    }


def find_best_edge_placement(
    edge_piece: PuzzlePiece,
    piece_shapes: Dict[int, np.ndarray],
    current_placements: List[dict],
    target: np.ndarray,
    current_score: float,
    renderer,
    scorer,
    all_guesses,
    all_scores,
) -> Optional[dict]:
    """
    Smart edge placement:
    1. Try each side (right, left, top, bottom) with all 4 rotations
    2. If side doesn't improve score -> abort that side immediately
    3. If side improves -> slide along axis to find best position
    """

    piece_id = int(edge_piece.id)
    height, width = target.shape

    # Define sides to try - try all 4 rotations per side
    sides = [
        "right",
        "left",
        "top",
        "bottom",
    ]

    rotations = set()
    # 2) Add primary rotation if this piece belongs on this side
    if edge_piece.primary_edge_rotation is not None:
        rotations.add(edge_piece.primary_edge_rotation)
        rotations.add((edge_piece.primary_edge_rotation + 90) % 360)
        rotations.add((edge_piece.primary_edge_rotation + 180) % 360)
        rotations.add((edge_piece.primary_edge_rotation + 270) % 360)
    # 3) Fallback safety net
    if not rotations:
        rotations = {0, 90, 180, 270}
    rotations = list(rotations)

    best_placement = None
    best_score = current_score

    for side_name in sides:
        print(f"        Trying {side_name} side...")

        side_best_score = current_score
        side_best_placement = None

        # Try all rotations for this side
        for rotation in rotations:
            rotated_mask = rotate_and_crop(piece_shapes[piece_id], rotation)
            piece_h, piece_w = rotated_mask.shape

            # Initial test position for this side
            if side_name == "right":
                test_x = width - piece_w
                test_y = (height - piece_h) / 2
                axis_type = "vertical"
            elif side_name == "left":
                test_x = 0
                test_y = (height - piece_h) / 2
                axis_type = "vertical"
            elif side_name == "bottom":
                test_x = (width - piece_w) / 2
                test_y = height - piece_h
                axis_type = "horizontal"
            else:  # top
                test_x = (width - piece_w) / 2
                test_y = 0
                axis_type = "horizontal"

            # Test initial position
            test_placement = {
                "piece_id": piece_id,
                "x": test_x,
                "y": test_y,
                "theta": rotation,
                "side": side_name,
            }

            test_placements = current_placements + [test_placement]

            # ADD TEST TO VISUALIZER
            all_guesses.append(test_placements.copy())

            rendered = renderer.render(test_placements, piece_shapes)
            test_score = scorer.score(rendered, target)
            all_scores.append(test_score)

            if test_score > side_best_score:
                side_best_score = test_score
                side_best_placement = test_placement.copy()
                side_best_placement["axis_type"] = axis_type

        print(
            f"          Best rotation: θ={side_best_placement['theta'] if side_best_placement else 'N/A'}°, score={side_best_score:.1f}"
        )

        # If no improvement on this side, skip sliding
        if side_best_score <= current_score:
            print(f"          → No improvement, skip {side_name}")
            continue

        # This side improved! Slide along axis to optimize
        print(f"          → {side_name} improved! Sliding to optimize...")

        optimized = slide_along_axis(
            piece_id=piece_id,
            piece_shapes=piece_shapes,
            current_placements=current_placements,
            target=target,
            initial_placement=side_best_placement,
            axis_type=side_best_placement["axis_type"],
            side_name=side_name,
            renderer=renderer,
            scorer=scorer,
            all_guesses=all_guesses,
            all_scores=all_scores,
        )

        if optimized["score"] > best_score:
            best_score = optimized["score"]
            best_placement = optimized["placement"]
            print(f"          → Optimized score: {best_score:.1f}")

    return best_placement


def slide_along_axis(
    piece_id: int,
    piece_shapes: Dict[int, np.ndarray],
    current_placements: List[dict],
    target: np.ndarray,
    initial_placement: dict,
    axis_type: str,
    side_name: str,
    renderer,
    scorer,
    all_guesses,
    all_scores,
) -> dict:
    """
    Slide piece along its axis (vertical or horizontal) to find best position.
    Uses a simple grid search with 20 test positions.

    IMPORTANT: Adds each test position to visualizer!
    """

    height, width = target.shape
    rotated_mask = rotate_and_crop(
        piece_shapes[piece_id], initial_placement["theta"]
    )
    piece_h, piece_w = rotated_mask.shape

    best_placement = initial_placement.copy()
    best_score = -float("inf")

    # Determine search range
    if axis_type == "vertical":
        # Slide up/down (vary y)
        positions = np.linspace(0, max(0, height - piece_h), num=20)

        for y_pos in positions:
            test_placement = initial_placement.copy()
            test_placement["y"] = float(y_pos)

            test_placements = current_placements + [test_placement]
            rendered = renderer.render(test_placements, piece_shapes)
            score = scorer.score(rendered, target)

            # ADD TO VISUALIZER
            all_guesses.append(test_placements.copy())
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_placement = test_placement.copy()

    else:  # horizontal
        # Slide left/right (vary x)
        positions = np.linspace(0, max(0, width - piece_w), num=20)

        for x_pos in positions:
            test_placement = initial_placement.copy()
            test_placement["x"] = float(x_pos)

            test_placements = current_placements + [test_placement]
            rendered = renderer.render(test_placements, piece_shapes)
            score = scorer.score(rendered, target)

            # ADD TO VISUALIZER
            all_guesses.append(test_placements.copy())
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_placement = test_placement.copy()

    return {"placement": best_placement, "score": best_score}
