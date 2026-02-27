#!/usr/bin/env python3
"""
Einstiegspunkt fuer den Roboter.

Liest Puzzleteile aus ./input, loest das Puzzle und gibt die
Bewegungsdaten (x, y, rotation) pro Teil als JSON auf stdout aus.

Verwendung:
  python robot.py
  python robot.py --positions '[{"piece_id":0,"x":100,"y":200}, ...]'
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.core.config import Config
from src.solver.iterative_solver import IterativeSolver
from src.solver.movement_analyzer import MovementAnalyzer
from src.solver.piece_analyzer import PieceAnalyzer
from src.solver.validation.scorer import PlacementScorer
from src.ui.simulator.guess_renderer import GuessRenderer
from src.solver.guess_generator import GuessGenerator
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece
from src.vision.mock_puzzle_creator import MockPuzzleGenerator


INPUT_DIR = Path(__file__).parent / "input"
WORK_DIR = Path("data/robot_pieces")
SCORE_THRESHOLD = 205000.0


def copy_pieces():
    """Kopiere Puzzleteile von /input in lokales Arbeitsverzeichnis."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    pieces = sorted(INPUT_DIR.glob("piece_*.png"))
    if not pieces:
        print(json.dumps({"error": f"Keine piece_*.png Dateien in {INPUT_DIR} gefunden"}))
        sys.exit(1)

    for src in pieces:
        shutil.copy2(src, WORK_DIR / src.name)

    return sorted(WORK_DIR.glob("piece_*.png"))


def load_pieces(piece_paths):
    """Lade Puzzleteile und erstelle PuzzlePiece-Objekte."""
    puzzle_pieces = []
    piece_shapes = {}
    piece_ids = []

    a5_width = 840
    a5_height = 594
    margin = 80

    corner_positions = [
        (margin, margin),
        (a5_width - margin, margin),
        (margin, a5_height - margin),
        (a5_width - margin, a5_height - margin),
    ]

    for idx, path in enumerate(piece_paths):
        piece_id = int(path.stem.split("_")[1])

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Maske extrahieren
        if img.shape[2] == 4:
            mask = img[:, :, 3]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = (mask > 127).astype(np.uint8)

        piece_shapes[idx] = mask
        piece_ids.append(idx)

        # Pick-Position zuweisen
        piece_h, piece_w = img.shape[:2]
        corner_idx = idx % len(corner_positions)
        base_x, base_y = corner_positions[corner_idx]
        x = max(margin, min(a5_width - piece_w - margin, base_x))
        y = max(margin, min(a5_height - piece_h - margin, base_y))

        piece = PuzzlePiece(pid=str(piece_id), pick=Pose(x=float(x), y=float(y), theta=0.0))
        puzzle_pieces.append(piece)

    return piece_ids, piece_shapes, puzzle_pieces


def create_surface_layout():
    """Erstelle Source/Target Layout (identisch zu pipeline.py)."""
    target_width, target_height = 420, 594
    source_width, source_height = 840, 594
    padding = 100

    global_width = source_width + target_width + padding * 3
    global_height = max(source_height, target_height) + padding * 2

    source_offset_x = padding
    source_offset_y = (global_height - source_height) // 2
    target_offset_x = source_width + padding * 2
    target_offset_y = (global_height - target_height) // 2

    return {
        "global": {"width": global_width, "height": global_height},
        "source": {
            "width": source_width, "height": source_height,
            "offset_x": source_offset_x, "offset_y": source_offset_y,
            "mask": np.ones((source_height, source_width), dtype=np.uint8),
        },
        "target": {
            "width": target_width, "height": target_height,
            "offset_x": target_offset_x, "offset_y": target_offset_y,
            "mask": np.ones((target_height, target_width), dtype=np.uint8),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Puzzle-Solver fuer Roboter")
    parser.add_argument(
        "--positions",
        type=str,
        help='JSON-Array mit Pick-Positionen: [{"piece_id":0,"x":100,"y":200}, ...]',
    )
    args = parser.parse_args()

    # 1. Teile kopieren und laden
    piece_paths = copy_pieces()
    piece_ids, piece_shapes, puzzle_pieces = load_pieces(piece_paths)

    # Positionen aus Argument uebernehmen
    if args.positions:
        try:
            positions = json.loads(args.positions)
            for pos in positions:
                pid = str(pos["piece_id"])
                for piece in puzzle_pieces:
                    if piece.id == pid:
                        piece.pick_pose = Pose(
                            x=float(pos["x"]) * 2.0,
                            y=float(pos["y"]) * 2.0,
                            theta=0.0,
                        )
                        break
        except (json.JSONDecodeError, KeyError) as e:
            print(json.dumps({"error": f"Ungueltige --positions: {e}"}))
            sys.exit(1)

    if not piece_ids:
        print(json.dumps({"error": "Keine gueltigen Puzzleteile geladen"}))
        sys.exit(1)

    # 2. Teile analysieren
    PieceAnalyzer.analyze_all_pieces(puzzle_pieces, piece_shapes)

    # 3. Puzzle loesen
    surfaces = create_surface_layout()
    target = surfaces["target"]["mask"]

    renderer = GuessRenderer(width=target.shape[1], height=target.shape[0])
    scorer = PlacementScorer(overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5)
    guess_generator = GuessGenerator(rotation_step=90)

    solver = IterativeSolver(renderer=renderer, scorer=scorer, guess_generator=guess_generator)
    solution = solver.solve_iteratively(
        piece_shapes=piece_shapes,
        target=target,
        puzzle_pieces=puzzle_pieces,
        score_threshold=SCORE_THRESHOLD,
    )

    if not solution.success:
        print(json.dumps({"error": "Keine Loesung gefunden", "score": solution.score}))
        sys.exit(1)

    # Place-Pose auf PuzzlePiece-Objekte setzen
    for placement in solution.remaining_placements:
        pid = placement["piece_id"]
        for piece in puzzle_pieces:
            if int(piece.id) == pid:
                piece.place_pose = Pose(x=placement["x"], y=placement["y"], theta=placement["theta"])
                break

    # 4. Bewegungsdaten berechnen
    best_guess = solution.remaining_placements
    movement_data = MovementAnalyzer.analyze_best_solution_movements(
        puzzle_pieces=puzzle_pieces,
        best_guess=best_guess,
        piece_shapes=piece_shapes,
        surfaces=surfaces,
    )

    movements = movement_data.get("movements", {})

    # 5. Ausgabe: JSON-Array mit Bewegungen pro Teil
    output = []
    for piece in sorted(puzzle_pieces, key=lambda p: int(p.id)):
        piece_id = int(piece.id)
        if piece_id in movements:
            m = movements[piece_id]
            output.append({
                "piece_id": piece_id,
                "x_mm": round(m["x_mm"], 2),
                "y_mm": round(m["y_mm"], 2),
                "rotation": round(m["rotation"], 2),
            })
        else:
            output.append({
                "piece_id": piece_id,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "rotation": 0.0,
            })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
