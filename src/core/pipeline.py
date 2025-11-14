# src/core/pipeline.py

"""
Haupt-Pipeline orchestriert alle Schritte
"""
from dataclasses import dataclass
import os
from time import time
from typing import Optional

import cv2

from src.solver.piece_analyzer import PieceAnalyzer
from .config import Config
from ..utils.logger import setup_logger
from ..solver.guess_generator import GuessGenerator 
from ..ui.simulator.guess_renderer import GuessRenderer
from ..solver.validation.scorer import PlacementScorer
import numpy as np


@dataclass
class PipelineResult:
    """Ergebnis der Pipeline"""
    success: bool
    duration: float
    message: str
    solution: Optional[dict] = None


class PuzzlePipeline:
    """
    Haupt-Pipeline fuer Puzzle-Loesung
    
    Schritte:
    1. Bildaufnahme & Preprocessing
    2. Segmentierung
    3. Feature-Extraktion
    4. Puzzle loesen
    5. Validierung
    6. (PREN2) Hardware-Steuerung
    """

    def __init__(self, config: Config, show_ui: bool = False):
        self.config = config
        self.logger = setup_logger("pipeline")
        self.show_ui = show_ui
        
        # Initialize solver components - renderer will be created with target
        self.guess_generator = GuessGenerator(rotation_step=90)
        self.renderer = None  # Will be created after we have target
        self.scorer = PlacementScorer(
            overlap_penalty=2.0,
            coverage_reward=1.0,
            gap_penalty=0.5
        )
        
    def run(self) -> PipelineResult:
        """Fuehrt die komplette Pipeline aus"""
        self.logger.info("Pipeline gestartet...")
        start_time = time()
        
        try:
            # Phase 1: Vision
            self.logger.info("Phase 1: Bildverarbeitung")
            pieces, piece_shapes, corner_info = self._process_vision()
            
            # Phase 2: Solving
            self.logger.info("Phase 2: Puzzle loesen")
            solution = self._solve_puzzle(pieces, piece_shapes, corner_info)
            
            # Phase 3: Validation
            self.logger.info("Phase 3: Validierung")
            is_valid = self._validate_solution(solution)
            
            # Launch UI even if validation failed (for debugging)
            if self.show_ui and solution:
                self._launch_ui(solution)
            
            if not is_valid:
                return PipelineResult(
                    success=False,
                    duration=time() - start_time,
                    message="Loesung konnte nicht validiert werden",
                    solution=solution  # Still return solution for debugging
                )
            
            # Phase 4: Hardware (nur PREN2)
            if self.config.hardware.enabled:
                self.logger.info("Phase 4: Hardware-Steuerung")
                self._execute_hardware(solution)
            
            duration = time() - start_time
            self.logger.info(f"✓ Pipeline erfolgreich abgeschlossen ({duration:.2f}s)")
            
            return PipelineResult(
                success=True,
                duration=duration,
                message="Puzzle erfolgreich geloest",
                solution=solution
            )
            
        except Exception as e:
            self.logger.exception(f"Pipeline-Fehler: {e}")
            return PipelineResult(
                success=False,
                duration=time() - start_time,
                message=f"Fehler: {str(e)}"
            )
    
    def _process_vision(self):
        """Bildverarbeitung"""
        self.logger.info("  → Bildaufnahme...")
        
        from ..vision.mock_puzzle_creator import MockPuzzleGenerator
        
        generator = MockPuzzleGenerator(output_dir="data/mock_pieces")
        
        # Check if we already have saved pieces
        all_piece_files = list(generator.output_dir.glob("piece_*.png"))
        existing_pieces = [p for p in all_piece_files if not p.stem.endswith('_corners')]
        
        if not existing_pieces or self.config.vision.regenerate_mock:
            self.logger.info("  → Generiere Mock-Puzzle...")
            
            # Generate new puzzle
            full_image, piece_images, debug_image = generator.generate_puzzle()
            
            # Save pieces
            piece_paths = generator.save_pieces(piece_images)
            
            # Save debug image
            cv2.imwrite("data/mock_pieces/debug_cuts.png", debug_image)
            self.logger.info("  → Mock-Puzzle gespeichert in data/mock_pieces/")
        else:
            self.logger.info(f"  → Lade {len(existing_pieces)} existierende Mock-Teile...")
        
        self.logger.info("  → Segmentierung...")
        self.logger.info("  → Feature-Extraktion...")
        
        # Load pieces for solver
        piece_ids, piece_shapes = generator.load_pieces_for_solver()
        
        # ANALYZE ALL PIECES FOR CORNERS
        piece_corner_info = PieceAnalyzer.analyze_all_pieces(piece_shapes)
        
        # Save visualizations
        for piece_id, info in piece_corner_info.items():
            if piece_id in piece_shapes:
                vis = PieceAnalyzer.visualize_corners(
                    (piece_shapes[piece_id] * 255).astype(np.uint8),
                    info
                )
                debug_dir = "data/mock_pieces/debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/piece_{piece_id}_corners.png", vis)
        
        self.logger.info(f"  → {len(piece_ids)} Teile geladen und analysiert")
        
        return piece_ids, piece_shapes, piece_corner_info  # Return corner info too!

    def _solve_puzzle(self, pieces, piece_shapes, piece_corner_info):
        """Puzzle loesen mit iterativem Ansatz"""
        self.logger.info("  → Layout berechnen...")
        
        # Create target with exact dimensions
        target = self._create_target_layout(len(pieces))
        
        self.logger.info(f"  → Target dimensions: {target.shape}")
        
        # Create renderer to match target dimensions
        height, width = target.shape
        self.renderer = GuessRenderer(width=width, height=height)
        
        self.logger.info("  → Iteratives Loesen starten...")
        
        from ..solver.iterative_solver import IterativeSolver
        
        iterative_solver = IterativeSolver(
            renderer=self.renderer,
            scorer=self.scorer,
            guess_generator=self.guess_generator
        )
        
        # Solve iteratively, trying different anchor pieces
        solution = iterative_solver.solve_iteratively(
            piece_shapes=piece_shapes,
            piece_corner_info=piece_corner_info,
            target=target,
            score_threshold=230000.0
        )
        
        if not solution.success:
            self.logger.warning("  ! Keine gute Loesung gefunden")
        else:
            self.logger.info(f"  ✓ Loesung gefunden mit Score: {solution.score:.2f}")
        
        # NEW: Use ALL guesses collected during solving
        all_guesses = solution.all_guesses if solution.all_guesses else []
        
        # Add final solution if it's not already in the list
        if solution.remaining_placements:
            if not all_guesses or all_guesses[-1] != solution.remaining_placements:
                all_guesses.append(solution.remaining_placements)
        
        self.logger.info(f"  → Collected {len(all_guesses)} guesses for visualization")
        
        # Find best guess and its index
        best_score = solution.score
        best_guess = solution.remaining_placements
        best_guess_index = len(all_guesses) - 1 if all_guesses else 0
        
        return {
            'placements': solution.remaining_placements,
            'score': solution.score,
            'rendered': None,
            'target': target,
            'guesses': all_guesses,
            'piece_shapes': piece_shapes,
            'best_score': best_score,
            'best_guess': best_guess,
            'best_guess_index': best_guess_index,
            'renderer': self.renderer  # Pass renderer to UI
        }
    
    def _create_target_layout(self, num_pieces):
        """Erstelle Ziel-Layout basierend auf Anzahl Teile"""
        
        width = 420
        height = 594
        # Create target with EXACT dimensions needed
        target = np.ones((height, width), dtype=np.uint8)
        
        target_area = np.sum(target > 0)
        self.logger.info(f"  → Target: {width}x{height}, area={target_area}")
        
        # DEBUG: Check if target is actually filled
        if np.sum(target) == 0:
            self.logger.error("  ❌ TARGET IS EMPTY!")
        
        return target


    def _validate_solution(self, solution):
        """Loesung validieren"""
        self.logger.info("  → Geometrie pruefen...")
        
        # Check if we have a solution
        if not solution or 'score' not in solution:
            return False
        
        # Check if score is reasonable - ADJUST THRESHOLD
        if solution['score'] < -10000:  # Less strict for now
            self.logger.warning(f"  ! Score zu niedrig: {solution['score']}")
            return False
        
        self.logger.info("  → Konfidenz berechnen...")
        # Better confidence calculation
        max_possible_score = 90000  # Approximate max coverage
        confidence = min(100, max(0, (solution['score'] + 10000) / max_possible_score * 100))
        self.logger.info(f"  → Konfidenz: {confidence:.1f}%")
        
        return True
    
    def _execute_hardware(self, solution):
        """Hardware ansteuern (PREN2)"""
        self.logger.info("  → Motoren initialisieren...")
        self.logger.info("  → Teile platzieren...")
        
        # Print placements for hardware
        for placement in solution['placements']:
            self.logger.info(
                f"    Piece {placement['piece_id']}: "
                f"x={placement['x']:.1f}, y={placement['y']:.1f}, "
                f"theta={placement['theta']:.1f}"
            )
        
        # TODO: Implementierung in PREN2
        pass
    
    def _launch_ui(self, solution):
        """Launch Kivy UI to visualize the solution."""
        self.logger.info("🎬 Starte Visualisierung...")
        
        from ..ui.simulator.solver_visualizer import SolverVisualizerApp
        
        solver_data = {
            'guesses': solution['guesses'],
            'piece_shapes': solution['piece_shapes'],
            'target': solution['target'],
            'best_score': solution['best_score'],
            'best_guess': solution.get('best_guess'),
            'best_guess_index': solution.get('best_guess_index', 0),
            'renderer': solution['renderer']  # Add the renderer
        }
        
        self.logger.info(f"  → Passing {len(solution['guesses'])} guesses to visualizer")
        
        app = SolverVisualizerApp(solver_data)
        app.run()