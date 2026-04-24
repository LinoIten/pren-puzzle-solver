"""
Zentrale Konfiguration für das Puzzle-Solver System
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VisionConfig:
    """Konfiguration für Bildverarbeitung"""

    camera_id: int = 0
    image_width: int = 1920
    image_height: int = 1080
    threshold_value: int = 127
    min_contour_area: int = 1000
    regenerate_mock: bool = False
    num_cuts: int | None = None  # None = zufällig (2 oder 3); 3 = 6 Teile


@dataclass
class SolverConfig:
    """Konfiguration für Puzzle-Solver"""

    max_solve_time: float = 90.0  # 1.5 Minute
    rotation_step: int = 15  # Rotation increment in degrees
    coarse_rotation_step: int = 45  # Coarser for initial searchn


@dataclass
class ResolutionConfig:
    """Globale Aufloesung fuer den Solver (Performance-Tuning).

    scale=1.0 entspricht 2 px/mm (A4 = 420x594, A5 = 840x594).
    Kleinere Werte beschleunigen das Scoring quadratisch (weniger Pixel),
    reduzieren aber die Praezision. Die Scorer-Gewichte werden automatisch
    mit 1/scale^2 multipliziert, damit `score_threshold` unabhaengig von
    der Aufloesung vergleichbar bleibt.
    """

    scale: float = 0.5

    # Basis-Dimensionen bei scale=1.0 (2 Pixel pro mm)
    base_a4_width: int = 420
    base_a4_height: int = 594
    base_a5_width: int = 840
    base_a5_height: int = 594

    @property
    def a4_width(self) -> int:
        return max(1, int(round(self.base_a4_width * self.scale)))

    @property
    def a4_height(self) -> int:
        return max(1, int(round(self.base_a4_height * self.scale)))

    @property
    def a5_width(self) -> int:
        return max(1, int(round(self.base_a5_width * self.scale)))

    @property
    def a5_height(self) -> int:
        return max(1, int(round(self.base_a5_height * self.scale)))

    @property
    def score_weight_multiplier(self) -> float:
        """Skalierungsfaktor fuer Scorer-Gewichte: 1/scale^2."""
        return 1.0 / (self.scale**2)


@dataclass
class SolverTuning:
    """Zentrale Tuning-Parameter fuer den Solver - alle an einem Ort."""

    # --- Scoring (scorer.py) ---
    overlap_penalty: float = 2.0
    coverage_reward: float = 1.0
    gap_penalty: float = 0.5
    score_threshold: float = 210000.0

    # --- Corner Detection (corner_detector.py) ---
    corner_angle_tolerance: int = 6  # Grad Abweichung von 90°
    corner_min_straightness: float = 0.9
    corner_min_edge_length: int = 25  # Pixel
    corner_min_quality: float = 0.65
    corner_max_overhang: int = 20  # Pixel
    corner_min_extent: int = 60  # Pixel
    corner_contour_epsilon: float = 0.01  # Anteil des Umfangs

    # --- Edge Detection (edge_detector.py) ---
    edge_min_length: int = 10  # Pixel
    edge_min_straightness: float = 0.8
    edge_min_score: float = 0.35
    edge_contour_epsilon: float = 0.008  # Anteil des Umfangs

    # --- Piece Classification (piece_analyzer.py) ---
    classify_corner_threshold: float = 0.85
    classify_edge_threshold: float = 0.8

    # --- Corner Fitter (corner_fitter.py) ---
    fitter_coarse_step: int = 5  # Grad
    fitter_fine_step: float = 0.5  # Grad
    fitter_fine_range: float = 10.0  # ±Grad um besten Winkel
    fitter_outside_limit: int = 100  # Pixel bevor Strafe
    fitter_edge_touch_bonus: int = 50000
    fitter_outside_penalty: int = 100
    fitter_edge_touch_distance: int = 10  # Pixel

    # --- Iterative Solver (iterative_solver.py) ---
    initial_corner_count: int = 60
    max_corners_to_refine: int = 20
    max_iterations: int = 600

    # --- Edge Placement (edge_placement.py) ---
    slide_positions: int = 20  # Gitterpositionen pro Achse
    center_piece_margin: int = 50  # Pixel vom Rand

    def scaled(self, resolution_scale: float) -> "SolverTuning":
        """Gibt eine Kopie zurueck, bei der alle Pixel-basierten Parameter mit
        resolution_scale skaliert sind. Wird von der Pipeline benutzt, damit
        Corner-/Edge-Detection auch bei niedrigerer Aufloesung korrekt arbeitet.
        Schwellwerte ohne Pixelbezug (Winkel, Verhaeltnisse) bleiben unveraendert.
        """
        from copy import copy

        s = resolution_scale
        t = copy(self)

        def px(v: int) -> int:
            return max(1, int(round(v * s)))

        t.corner_min_edge_length = px(self.corner_min_edge_length)
        t.corner_max_overhang = px(self.corner_max_overhang)
        t.corner_min_extent = px(self.corner_min_extent)
        t.edge_min_length = px(self.edge_min_length)
        t.fitter_outside_limit = px(self.fitter_outside_limit)
        t.fitter_edge_touch_distance = px(self.fitter_edge_touch_distance)
        t.center_piece_margin = px(self.center_piece_margin)

        return t


@dataclass
class HardwareConfig:
    """Konfiguration für Hardware (PREN2)"""

    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    enabled: bool = False  # In PREN1 deaktiviert


@dataclass
class Config:
    """Haupt-Konfiguration"""

    vision: VisionConfig = field(default_factory=VisionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    tuning: SolverTuning = field(default_factory=SolverTuning)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)

    # Pfade
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        # Pfade setzen
        self.data_dir = self.project_root / "data"
        self.output_dir = self.data_dir / "results"

        # Verzeichnisse erstellen falls nicht vorhanden
        self.output_dir.mkdir(parents=True, exist_ok=True)
