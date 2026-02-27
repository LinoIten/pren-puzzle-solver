"""
Corner placement logic for the iterative solver.
Handles corner combination generation, evaluation, and placement.
"""

import itertools

import numpy as np

from src.utils.geometry import rotate_and_crop


def generate_corner_combinations(corner_candidates):
    """Generate all possible corner combinations for given candidates, sorted by quality."""
    # For puzzles, we need exactly 4 corners
    if len(corner_candidates) < 4:
        print(f"  ❌ Not enough corner candidates: {len(corner_candidates)} < 4")
        return []

    # Step 1: Permutations of pieces (which piece in which corner)
    # We need exactly 4 pieces for the 4 corners
    piece_permutations = list(itertools.permutations(corner_candidates, 4))

    print(
        f"  Piece permutations: {len(piece_permutations)} (which piece → which corner)"
    )

    # Step 2: For each permutation, generate corner rotation combinations
    all_corner_combinations = []

    for perm in piece_permutations:
        # For this permutation, get all rotation combinations
        piece_corner_options = [[i for i in range(len(p.corners))] for p in perm]
        rotation_combos = list(itertools.product(*piece_corner_options))

        # Store (piece_permutation, rotation_combo)
        for rotation_combo in rotation_combos:
            all_corner_combinations.append((perm, rotation_combo))

    # Sort by quality (sum of corner qualities for each combo)
    def combo_quality(combo):
        perm, rotation_indices = combo
        total_quality = 0
        for piece, corner_idx in zip(perm, rotation_indices):
            if corner_idx < len(piece.corners):
                total_quality += piece.corners[corner_idx].quality
        return total_quality

    all_corner_combinations.sort(key=combo_quality, reverse=True)
    return all_corner_combinations


def place_corners(corner_pieces, piece_rotations, piece_shapes, target):
    """Place corner pieces in the 4 corners of the target."""
    height, width = target.shape

    corners = [
        ("bottom_right", width, height, 0),
        ("bottom_left", 0, height, 270),
        ("top_left", 0, 0, 180),
        ("top_right", width, 0, 90),
    ]

    placements = []

    for piece, (corner_name, corner_x, corner_y, rotation_offset) in zip(
        corner_pieces, corners[: len(corner_pieces)]
    ):
        piece_id = int(piece.id)
        rotation = piece_rotations[piece_id] + rotation_offset

        rotated = rotate_and_crop(piece_shapes[piece_id], rotation)
        piece_h, piece_w = rotated.shape

        if corner_name == "top_left":
            x, y = 0, 0
        elif corner_name == "top_right":
            x, y = width - piece_w, 0
        elif corner_name == "bottom_left":
            x, y = 0, height - piece_h
        else:  # bottom_right
            x, y = width - piece_w, height - piece_h

        placements.append(
            {
                "piece_id": piece_id,
                "x": float(x),
                "y": float(y),
                "theta": float(rotation),
            }
        )

    return placements


def evaluate_corner_layouts(
    all_combinations,
    initial_corner_count,
    renderer,
    scorer,
    piece_shapes,
    target,
    all_guesses,
    all_scores,
):
    """
    Evaluate corner-only layouts (Phase 1).

    Returns list of (combo_idx, piece_perm, rotation_indices, placements, score)
    sorted by score descending.
    """
    initial_corners_to_evaluate = min(initial_corner_count, len(all_combinations))
    print(
        f"  Will evaluate {initial_corners_to_evaluate} corner layouts before trying edges..."
    )

    corner_evaluations = []

    for combo_idx in range(initial_corners_to_evaluate):
        piece_permutation, rotation_indices = all_combinations[combo_idx]

        # Build rotations for this specific piece arrangement
        piece_rotations = {}
        for piece, corner_idx in zip(piece_permutation, rotation_indices):
            piece_rotations[int(piece.id)] = piece.corners[
                corner_idx
            ].rotation_to_align

        # Place corners using this permutation
        corner_placements = place_corners(
            piece_permutation, piece_rotations, piece_shapes, target
        )

        # Score corner-only
        rendered = renderer.render(corner_placements, piece_shapes)
        score = scorer.score(rendered, target)

        # Store: (combo_idx, piece_perm, rotation_indices, placements, score)
        corner_evaluations.append(
            (
                combo_idx,
                piece_permutation,
                rotation_indices,
                corner_placements,
                score,
            )
        )

        # ADD CORNER-ONLY PLACEMENT TO VISUALIZER
        all_guesses.append(corner_placements)
        all_scores.append(score)

        if (combo_idx + 1) % 25 == 0:
            # Show which pieces are where
            piece_ids = [int(p.id) for p in piece_permutation]
            print(
                f"    Evaluated {combo_idx + 1}/{initial_corners_to_evaluate} corners... (e.g. pieces {piece_ids})"
            )

    # Sort by corner score
    corner_evaluations.sort(key=lambda x: x[4], reverse=True)  # x[4] is score

    print(f"\n  📊 Top 10 corner layouts (corner-only scores):")
    for i, (idx, piece_perm, rotation_indices, _, score) in enumerate(
        corner_evaluations[:10]
    ):
        piece_ids = [int(p.id) for p in piece_perm]
        print(f"    {i + 1}. Combo {idx}: pieces {piece_ids}, score={score:.1f}")

    return corner_evaluations
