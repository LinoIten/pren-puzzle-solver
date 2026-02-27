# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated puzzle-solving system for PREN HS25/FS26 (HSLU). Generates or loads puzzle pieces, analyzes geometric features (corners, edges), solves placement using an iterative algorithm, and visualizes the solution. Hardware integration (robot arm) is planned for PREN2.

## Commands

```bash
# Setup
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run main pipeline (with Kivy visualizer)
python main.py

# Run batch GUI controller (generation + solving + statistics)
python gui_controller.py

# Build standalone executable
./build.sh
```

No test runner or linter configured. Files in `tests/` are utility scripts, not unit tests.

## Architecture

**Pipeline flow:** `main.py` → `PuzzlePipeline` → Vision → Analysis → Solver → Validation → UI

### Core (`src/core/`)
- **`pipeline.py`** — `PuzzlePipeline` orchestrates all phases. Score threshold: 205,000.
- **`config.py`** — Dataclass-based config. No external config files.

### Vision (`src/vision/`)
- **`mock_puzzle_creator.py`** — `MockPuzzleGenerator` creates/loads pieces from `data/mock_pieces/`. Delegates cut generation to `cut_patterns.py`.
- **`cut_patterns.py`** — Standalone functions: `generate_wavy_cut()`, `generate_sharp_cut()`, `generate_square_cut()`.

### Solver (`src/solver/`)

Analysis:
- **`piece_analyzer.py`** — `PieceAnalyzer` classifies pieces as corner/edge/center. Delegates detection to `corner_detector.py` and `edge_detector.py`.
- **`corner_detector.py`** — `detect_corners()`, `calculate_corner_overhang()`.
- **`edge_detector.py`** — `detect_edges()`, `calculate_edge_rotations()`, straightness measurement.

Solving:
- **`iterative_solver.py`** — `IterativeSolver` orchestrates corner search + edge refinement. Delegates placement logic to `corner_placement.py` and `edge_placement.py`.
- **`corner_placement.py`** — `place_corners()`, `evaluate_corner_layouts()`, combination generation.
- **`edge_placement.py`** — `try_edge_placement_on_corners()`, `find_best_edge_placement()`, `slide_along_axis()`.
- **`corner_fitter.py`** — Fine rotation search (0.5° resolution) for corner alignment.
- **`guess_generator.py`** — Candidate placement generation.
- **`movement_analyzer.py`** — Piece movement data for visualization.
- **`validation/scorer.py`** — `PlacementScorer` (overlap, coverage, gap penalties).

### UI (`src/ui/simulator/`)
- **`solver_visualizer.py`** — Kivy-based interactive visualizer (play/pause/step). Delegates movement overlays to `movement_renderer.py`.
- **`movement_renderer.py`** — `MovementRenderer` draws COM dots, arrows, summary table, legend.
- **`guess_renderer.py`** — Renders placements in grayscale (scoring) or color (display).

### Utils (`src/utils/`)
- **`geometry.py`** — Shared `rotate_and_crop(shape, angle, crop=True)`. Used by solver, renderer, and analyzer modules.
- **`puzzle_piece.py`** — `PuzzlePiece` class + `CornerData`/`EdgeData` dataclasses.
- **`pose.py`** — `Pose(x, y, theta)` for 2D positioning.

### Hardware (`src/hardware/`)
- Robot interface stub. Disabled in PREN1 (`HardwareConfig.enabled = False`).

## Key design patterns

- **Delegation over inheritance:** Large classes delegate to extracted modules (e.g., `IterativeSolver` → `corner_placement` + `edge_placement`). Original methods kept as thin wrappers.
- **In-place enrichment:** `PieceAnalyzer.analyze_piece()` mutates `PuzzlePiece` objects directly (populates `.piece_type`, `.corners`, `.edges`).
- **Shared rotation utility:** All shape rotation goes through `src.utils.geometry.rotate_and_crop()`. The `crop=False` variant is used only by `corner_fitter.py`.
- **Visualizer data passing:** Pipeline builds a `solver_data` dict passed to `SolverVisualizer`. Guesses and scores are collected by reference (list mutation) during solving.

## Conventions

- Python 3.13, no type checker or formatter.
- Log messages and comments in **German**; identifiers in **English**.
- Imports: relative within `src/` (e.g., `from ..solver.guess_generator import GuessGenerator`), absolute for new extracted modules (e.g., `from src.utils.geometry import rotate_and_crop`).
- Key dependencies: OpenCV (headless), NumPy, SciPy, Kivy, Pillow, matplotlib.
