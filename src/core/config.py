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
    regenerate_mock: bool = True

@dataclass
class SolverConfig:
    """Konfiguration für Puzzle-Solver"""
    max_solve_time: float = 90.0  # 1.5 Minute
    rotation_step: int = 15  # Rotation increment in degrees
    coarse_rotation_step: int = 45  # Coarser for initial searchn

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
    
    # Pfade
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    
    def __post_init__(self):
        # Pfade setzen
        self.data_dir = self.project_root / "data"
        self.output_dir = self.data_dir / "results"
        
        # Verzeichnisse erstellen falls nicht vorhanden
        self.output_dir.mkdir(parents=True, exist_ok=True)
