"""
Loads puzzle pieces from the camera vision system output.

Expected input directory layout:
    input/
        parts.json      – metadata (px_per_mm, A4 dims, centroids)
        png_0.png       – binary mask for piece 0 (0/255, grayscale)
        png_1.png
        ...
"""

import json
import cv2
import numpy as np
from pathlib import Path

from src.utils.puzzle_piece import PuzzlePiece
from src.utils.pose import Pose


class CameraLoader:
    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)
        self._json: dict = {}

    # ------------------------------------------------------------------
    # Public API (mirrors MockPuzzleGenerator.load_pieces_for_solver)
    # ------------------------------------------------------------------

    def load_json(self) -> dict:
        """Read and return parts.json.  Raises if missing."""
        json_path = self.input_dir / "parts.json"
        with open(json_path, "r") as f:
            self._json = json.load(f)
        return self._json

    @property
    def px_per_mm(self) -> float:
        return float(self._json["px_per_mm"])

    @property
    def a4_width_mm(self) -> int:
        return round(self._json["a4_size_mm"]["width"])

    @property
    def a4_height_mm(self) -> int:
        return round(self._json["a4_size_mm"]["height"])

    def load_pieces_for_solver(self, scale: float = 1.0):
        """
        Load piece masks, apply scale, return (piece_ids, piece_shapes).

        scale = solver_px_per_mm / native_px_per_mm
        """
        piece_shapes: dict[int, np.ndarray] = {}
        piece_ids: list[int] = []

        for part in self._json["parts"]:
            idx = part["index"] - 1  # JSON is 1-based
            mask_path = self.input_dir / part["mask_filename"]

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  ⚠️  Mask not found: {mask_path}")
                continue

            mask = (mask > 127).astype(np.uint8)

            if scale != 1.0:
                new_h = max(1, int(round(mask.shape[0] * scale)))
                new_w = max(1, int(round(mask.shape[1] * scale)))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            piece_shapes[idx] = mask
            piece_ids.append(idx)

        return piece_ids, piece_shapes

    def create_puzzle_pieces(self, solver_px_per_mm: float) -> list:
        """
        Build PuzzlePiece objects with pick poses from centroid data.

        centroid_mm uses origin bottom-left, y-up.  We convert to image
        coordinates (origin top-left, y-down) for the solver canvas.
        """
        pieces = []
        height_mm = self.a4_height_mm

        for part in self._json["parts"]:
            idx = part["index"] - 1
            cx_mm = part["centroid_mm"]["x"]
            cy_mm = part["centroid_mm"]["y"]

            # Convert bottom-left origin → top-left origin
            x_px = cx_mm * solver_px_per_mm
            y_px = (height_mm - cy_mm) * solver_px_per_mm

            pick_pose = Pose(x=float(x_px), y=float(y_px), theta=0.0)
            piece = PuzzlePiece(pid=str(idx), pick=pick_pose)
            pieces.append(piece)

        return pieces

    @staticmethod
    def has_parts_json(directory: str | Path) -> bool:
        return (Path(directory) / "parts.json").exists()
