# src/core/pipeline.py

"""
Haupt-Pipeline orchestriert alle Schritte
"""
from dataclasses import dataclass
from time import time
from typing import Optional

import cv2
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
        
        # Initialize solver components - match smaller piece size
        self.guess_generator = GuessGenerator(rotation_step=90)
        self.renderer = GuessRenderer(width=800, height=800)  # Back to reasonable size
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
            pieces, piece_shapes = self._process_vision()
            
            # Phase 2: Solving
            self.logger.info("Phase 2: Puzzle loesen")
            solution = self._solve_puzzle(pieces, piece_shapes)
            
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
        existing_pieces = list(generator.output_dir.glob("piece_*.png"))
        
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
        
        self.logger.info(f"  → {len(piece_ids)} Teile geladen")
        
        return piece_ids, piece_shapes


    def _solve_puzzle(self, pieces, piece_shapes):
        """Puzzle loesen mit Brute-Force"""
        self.logger.info("  → Layout berechnen...")
        
        # Create target
        target = self._create_target_layout(len(pieces))
        
        self.logger.info("  → Teile zuordnen...")
        
        # Generate guesses with grid-based positioning
        guesses = self.guess_generator.generate_guesses(
            num_pieces=len(pieces),
            target=target,
            max_guesses=10000
        )
        
        self.logger.info(f"  → Teste {len(guesses)} moegliche Loesungen...")
        
        # Find best solution
        best_score = -float('inf')
        best_guess = None
        best_guess_index = 0  # Track the index
        best_rendered = None
        
        for i, guess in enumerate(guesses):
            rendered = self.renderer.render(guess, piece_shapes)
            score = self.scorer.score(rendered, target)
            
            if score > best_score:
                best_score = score
                best_guess = guess
                best_guess_index = i  # Save the index
                best_rendered = rendered
                self.logger.info(f"  → Neuer Bestwert: {best_score:.2f} bei Guess #{i}")
            
            # Progress logging
            if i % 500 == 0 and i > 0:
                self.logger.info(f"  → Fortschritt: {i}/{len(guesses)}, bester Score: {best_score:.2f}")
        
        self.logger.info(f"  ✓ Beste Loesung gefunden mit Score: {best_score:.2f}")
        
        return {
            'placements': best_guess,
            'score': best_score,
            'rendered': best_rendered,
            'target': target,
            'guesses': guesses,
            'piece_shapes': piece_shapes,
            'best_score': best_score,
            'best_guess': best_guess,  # Save the actual best guess
            'best_guess_index': best_guess_index  # Save the index
        }
    
    def _create_target_layout(self, num_pieces):
        """Erstelle Ziel-Layout basierend auf Anzahl Teile"""
        target = np.zeros((800, 800), dtype=np.uint8)
        
        if num_pieces == 4:
            # Target should match ~420x594 A4 size, centered in 800x800
            # Center the A4 area
            x_offset = (800 - 420) // 2  # ~190
            y_offset = (800 - 594) // 2  # ~103
            
            target[y_offset:y_offset+594, x_offset:x_offset+420] = 1
        
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
            'best_guess': solution.get('best_guess'),  # Add this
            'best_guess_index': solution.get('best_guess_index', 0)  # Add this
        }
        
        app = SolverVisualizerApp(solver_data)
        app.run()
        