# src/core/pipeline.py

"""
Haupt-Pipeline orchestriert alle Schritte
"""
from dataclasses import dataclass
import os
from time import time
from typing import Optional

import cv2

from src.solver.iterative_solver import IterativeSolver
from src.solver.piece_analyzer import PieceAnalyzer
from src.solver.movement_analyzer import calculate_movement_data_for_visualizer
from src.ui.simulator.solver_visualizer import SolverVisualizerApp
from .config import Config
from ..utils.logger import setup_logger
from ..solver.guess_generator import GuessGenerator 
from ..ui.simulator.guess_renderer import GuessRenderer
from ..solver.validation.scorer import PlacementScorer
import numpy as np
from src.utils.pose import Pose
from src.utils.puzzle_piece import PuzzlePiece
from ..vision.mock_puzzle_creator import MockPuzzleGenerator


@dataclass
class PipelineResult:
    """Ergebnis der Pipeline"""
    success: bool
    duration: float
    message: str
    solution: Optional[dict] = None

# constants
SCORE_THRESHOLD = 210000.0

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
            pieces, piece_shapes, corner_info, puzzle_pieces = self._process_vision()
            
            # Phase 2: Solving
            self.logger.info("Phase 2: Puzzle loesen")
            solution = self._solve_puzzle(pieces, piece_shapes, corner_info, puzzle_pieces)
            
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
        
        
        generator = MockPuzzleGenerator(output_dir="data/mock_pieces")
        
        # Check if we already have saved pieces
        all_piece_files = list(generator.output_dir.glob("piece_*.png"))
        existing_pieces = [p for p in all_piece_files if not p.stem.endswith('_corners')]
        
        puzzle_pieces = []
        
        if not existing_pieces or self.config.vision.regenerate_mock:
            self.logger.info("  → Generiere Mock-Puzzle...")
            
            # Generate new puzzle WITH positions - returns PuzzlePiece objects
            full_image, piece_images, debug_image, puzzle_pieces = generator.generate_puzzle_with_positions()
            
            # Save pieces
            piece_paths = generator.save_pieces(piece_images)
            
            # Save debug image
            cv2.imwrite("data/mock_pieces/debug_cuts.png", debug_image)
            self.logger.info("  → Mock-Puzzle gespeichert in data/mock_pieces/")
        else:
            self.logger.info(f"  → Lade {len(existing_pieces)} existierende Mock-Teile...")
            
            # A5 dimensions
            a5_width = 840
            a5_height = 594
            margin = 80
            
            corner_positions = [
                (margin, margin),
                (a5_width - margin, margin),
                (margin, a5_height - margin),
                (a5_width - margin, a5_height - margin)
            ]
            
            for idx, piece_path in enumerate(sorted(existing_pieces)):
                piece_id = int(piece_path.stem.split('_')[1])
                
                # Load to get dimensions
                img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    piece_h, piece_w = img.shape[:2]
                    
                    # Assign corner
                    corner_idx = idx % len(corner_positions)
                    base_x, base_y = corner_positions[corner_idx]
                    
                    # Clamp position
                    x = max(margin, min(a5_width - piece_w - margin, base_x))
                    y = max(margin, min(a5_height - piece_h - margin, base_y))
                    
                    pick_pose = Pose(x=float(x), y=float(y), theta=0.0)
                    piece = PuzzlePiece(pid=str(piece_id), pick=pick_pose)
                    puzzle_pieces.append(piece)
        
        self.logger.info("  → Segmentierung...")
        self.logger.info("  → Feature-Extraktion...")
        
        piece_ids, piece_shapes = generator.load_pieces_for_solver()

        PieceAnalyzer.analyze_all_pieces(puzzle_pieces, piece_shapes)

        # Print analysis results
        self.logger.info("\n" + "="*80)
        self.logger.info("PIECE ANALYSIS RESULTS")
        self.logger.info("="*80)

        corner_count = sum(1 for p in puzzle_pieces if p.piece_type == "corner")
        edge_count = sum(1 for p in puzzle_pieces if p.piece_type == "edge")
        center_count = sum(1 for p in puzzle_pieces if p.piece_type == "center")

        self.logger.info(f"\n📊 Classification:")
        self.logger.info(f"    Corner pieces: {corner_count}")
        self.logger.info(f"    Edge pieces: {edge_count}")
        self.logger.info(f"    Center pieces: {center_count}")

        # Save visualizations
        debug_dir = "data/mock_pieces/debug"
        os.makedirs(debug_dir, exist_ok=True)

        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            if piece_id in piece_shapes:
                self.logger.info(f"\n{piece.summary()}")
                vis = PieceAnalyzer.visualize_corners(piece_shapes[piece_id], piece)
                cv2.imwrite(f"{debug_dir}/piece_{piece_id}_analysis.png", vis)

        self.logger.info(f"\n  → {len(piece_ids)} Teile geladen und analysiert")
        self.logger.info("="*80 + "\n")

        # Return empty dict for backward compatibility
        return piece_ids, piece_shapes, {}, puzzle_pieces

    def _solve_puzzle(self, pieces, piece_shapes, piece_corner_info, puzzle_pieces):
        """Puzzle loesen mit iterativem Ansatz"""
        self.logger.info("  → Layout berechnen...")
        
        # Create surfaces with global coordinate system
        surfaces = self._create_surface_layout(len(pieces))
        target = surfaces['target']['mask']
        source = surfaces['source']['mask']
        
        self.logger.info(f"  → Global surface: {surfaces['global']['width']}x{surfaces['global']['height']}")
        self.logger.info(f"  → Target (A4) at ({surfaces['target']['offset_x']}, {surfaces['target']['offset_y']}): {surfaces['target']['width']}x{surfaces['target']['height']}")
        self.logger.info(f"  → Source (A5) at ({surfaces['source']['offset_x']}, {surfaces['source']['offset_y']}): {surfaces['source']['width']}x{surfaces['source']['height']}")
        
        height, width = target.shape
        self.renderer = GuessRenderer(width=width, height=height)
        
        iterative_solver = IterativeSolver(
            renderer=self.renderer,
            scorer=self.scorer,
            guess_generator=self.guess_generator
        )


        
        # Create initial placements from PuzzlePiece objects
        initial_placements = self._create_initial_placements_from_pieces(puzzle_pieces)
        
        # Solve iteratively - THIS CALL STAYS THE SAME
        solution = iterative_solver.solve_iteratively(
            piece_shapes=piece_shapes,
            target=target,
            puzzle_pieces=puzzle_pieces,  
            score_threshold=SCORE_THRESHOLD
        )
        if not solution.success:
            self.logger.warning("  ! Keine gute Loesung gefunden")
        else:
            self.logger.info(f"  ✓ Loesung gefunden mit Score: {solution.score:.2f}")
        
        # Solver already updated place_pose on PuzzlePiece objects
        for piece in puzzle_pieces:
            if piece.place_pose:
                self.logger.debug(f"  Piece {piece.id}: {piece.pick_pose} → {piece.place_pose}")
        
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
        
        # Populate place_pose on PuzzlePiece objects from solver results
        self.logger.info(f"  → Populating place_pose on {len(puzzle_pieces)} pieces")
        for placement in solution.remaining_placements:
            piece_id = placement['piece_id']
            # Find the matching PuzzlePiece object
            for piece in puzzle_pieces:
                if int(piece.id) == piece_id:
                    piece.place_pose = Pose(
                        x=placement['x'],
                        y=placement['y'],
                        theta=placement['theta']
                    )
                    piece.confidence = 1.0 if solution.score > SCORE_THRESHOLD else 0.5
                    self.logger.debug(f"    Piece {piece_id}: {piece.place_pose}")
                    break
        
        # Print movement instructions using PuzzlePiece objects
        self._print_movement_instructions_from_pieces(puzzle_pieces, surfaces)
        
        return {
            'placements': solution.remaining_placements,
            'score': solution.score,
            'rendered': None,
            'target': target,
            'source': source,
            'surfaces': surfaces,
            'initial_placements': initial_placements,
            'guesses': all_guesses,
            'piece_shapes': piece_shapes,
            'best_score': best_score,
            'best_guess': best_guess,
            'best_guess_index': best_guess_index,
            'renderer': self.renderer,
            'puzzle_pieces': puzzle_pieces  
        }
    
    def _create_surface_layout(self, num_pieces):
        """
        Erstelle globale Oberflaeche mit Source (A5) und Target (A4) Bereichen.
        
        Returns dict with:
            - global: {width, height}
            - source: {width, height, offset_x, offset_y, mask}
            - target: {width, height, offset_x, offset_y, mask}
        """
        
        # A4 target dimensions (at some scale, e.g., 2 pixels per mm)
        target_width = 420
        target_height = 594
        
        # A5 source dimensions (double the area of A4)
        # A4 area = 420 * 594 = 249,480 sq pixels
        # A5 should be ~500,000 sq pixels
        # Using: 840 x 594 (double width, same height for easier layout)
        source_width = 840
        source_height = 594
        
        # Global surface size (side by side with padding)
        padding = 100
        global_width = source_width + target_width + padding * 3
        global_height = max(source_height, target_height) + padding * 2
        
        # Calculate offsets (centered vertically)
        source_offset_x = padding
        source_offset_y = (global_height - source_height) // 2
        
        target_offset_x = source_width + padding * 2
        target_offset_y = (global_height - target_height) // 2
        
        # Create masks
        target_mask = np.ones((target_height, target_width), dtype=np.uint8)
        source_mask = np.ones((source_height, source_width), dtype=np.uint8)
        
        # Create global surface representation
        surfaces = {
            'global': {
                'width': global_width,
                'height': global_height
            },
            'source': {
                'width': source_width,
                'height': source_height,
                'offset_x': source_offset_x,
                'offset_y': source_offset_y,
                'mask': source_mask
            },
            'target': {
                'width': target_width,
                'height': target_height,
                'offset_x': target_offset_x,
                'offset_y': target_offset_y,
                'mask': target_mask
            }
        }
        
        return surfaces
    
    def _create_initial_placements_from_pieces(self, puzzle_pieces):
        """
        Create initial placements from PuzzlePiece objects.
        Uses pixel coordinates from pick_pose.
        """
        initial_placements = []
        
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            initial_placements.append({
                'piece_id': piece_id,
                'x': piece.pick_pose.x,
                'y': piece.pick_pose.y,
                'theta': piece.pick_pose.theta
            })
        
        self.logger.info(f"  → Created {len(initial_placements)} initial placements from PuzzlePiece objects")
        return initial_placements
    
    
    def _print_movement_instructions_from_pieces(self, puzzle_pieces, surfaces):
        """Print movement instructions using PuzzlePiece objects directly."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MOVEMENT INSTRUCTIONS (Global Coordinates)")
        self.logger.info("="*80)
        
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']
        target_offset_y = surfaces['target']['offset_y']
        
        for piece in puzzle_pieces:
            if piece.place_pose is None:
                self.logger.warning(f"  ! Piece {piece.id} has no place_pose")
                continue
            
            # Convert to global coordinates
            # Initial (pick) position: source area coordinates
            initial_global_x = source_offset_x + piece.pick_pose.x
            initial_global_y = source_offset_y + piece.pick_pose.y
            
            # Final (place) position: target area coordinates  
            final_global_x = target_offset_x + piece.place_pose.x
            final_global_y = target_offset_y + piece.place_pose.y
            
            # Calculate movement
            delta_x = final_global_x - initial_global_x
            delta_y = final_global_y - initial_global_y
            distance = np.sqrt(delta_x**2 + delta_y**2)
            
            # Rotation change
            rotation_change = (piece.place_pose.theta - piece.pick_pose.theta) % 360
            
            self.logger.info(f"\nPiece {piece.id}:")
            self.logger.info(f"  Pick:  {piece.pick_pose}")
            self.logger.info(f"  Place: {piece.place_pose}")
            self.logger.info(f"  Global pick:  ({initial_global_x:.1f}, {initial_global_y:.1f}) @ {piece.pick_pose.theta:.0f}°")
            self.logger.info(f"  Global place: ({final_global_x:.1f}, {final_global_y:.1f}) @ {piece.place_pose.theta:.0f}°")
            self.logger.info(f"  Movement: Δx={delta_x:.1f}, Δy={delta_y:.1f}, distance={distance:.1f}")
            if rotation_change != 0:
                self.logger.info(f"  Rotation: {rotation_change:.0f}°")
            
            # Direction
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                angle = np.degrees(np.arctan2(delta_y, delta_x))
                self.logger.info(f"  Direction: {angle:.1f}° from horizontal")
        
        self.logger.info("\n" + "="*80 + "\n")
    
    def _print_movement_instructions(self, initial_placements, final_placements, surfaces):
        """Print movement instructions for each piece in global coordinates."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MOVEMENT INSTRUCTIONS (Global Coordinates)")
        self.logger.info("="*80)
        
        # Create lookup for initial positions
        initial_lookup = {p['piece_id']: p for p in initial_placements}
        
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']
        target_offset_y = surfaces['target']['offset_y']
        
        for final in final_placements:
            piece_id = final['piece_id']
            
            if piece_id not in initial_lookup:
                self.logger.warning(f"  ! Piece {piece_id} not found in initial placements")
                continue
            
            initial = initial_lookup[piece_id]
            
            # Convert to global coordinates
            # Initial position: source area coordinates
            initial_global_x = source_offset_x + initial['x']
            initial_global_y = source_offset_y + initial['y']
            
            # Final position: target area coordinates  
            final_global_x = target_offset_x + final['x']
            final_global_y = target_offset_y + final['y']
            
            # Calculate movement
            delta_x = final_global_x - initial_global_x
            delta_y = final_global_y - initial_global_y
            distance = np.sqrt(delta_x**2 + delta_y**2)
            
            # Rotation change
            rotation_change = (final['theta'] - initial['theta']) % 360
            
            self.logger.info(f"\nPiece {piece_id}:")
            self.logger.info(f"  Initial (global): ({initial_global_x:.1f}, {initial_global_y:.1f}) @ {initial['theta']:.0f}°")
            self.logger.info(f"  Final (global):   ({final_global_x:.1f}, {final_global_y:.1f}) @ {final['theta']:.0f}°")
            self.logger.info(f"  Movement: Δx={delta_x:.1f}, Δy={delta_y:.1f}, distance={distance:.1f}")
            if rotation_change != 0:
                self.logger.info(f"  Rotation: {rotation_change:.0f}°")
            
            # Direction
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                angle = np.degrees(np.arctan2(delta_y, delta_x))
                self.logger.info(f"  Direction: {angle:.1f}° from horizontal")
        
        self.logger.info("\n" + "="*80 + "\n")

    def _validate_solution(self, solution):
        """Loesung validieren"""
        self.logger.info("  → Geometrie pruefen...")
        
        # Check if we have a solution
        if not solution or 'score' not in solution:
            return False
        
        if solution['score'] < -10000: 
            self.logger.warning(f"  ! Score zu niedrig: {solution['score']}")
            return False
        
        self.logger.info("  → Konfidenz berechnen...")
        # Better confidence calculation
        max_possible_score = 90000  
        confidence = min(100, max(0, (solution['score'] + 10000) / max_possible_score * 100))
        self.logger.info(f"  → Konfidenz: {confidence:.1f}%")
        
        return True
    
    def _execute_hardware(self, solution):
        """Hardware ansteuern (PREN2)"""
        self.logger.info("  → Motoren initialisieren...")
        self.logger.info("  → Teile platzieren...")
        
        # The movement instructions are already printed by _print_movement_instructions
        # Hardware can read them from the log
        
        # TODO: Implementierung in PREN2
        pass
    

    def _launch_ui(self, solution):
        """Launch Kivy UI to visualize the solution."""
        self.logger.info("🎬 Starte Visualisierung...")
        
        # Calculate movement data for best solution
        movement_data = None
        if solution.get('puzzle_pieces') and solution.get('best_guess'):
            movement_data = calculate_movement_data_for_visualizer(solution)
        
        solver_data = {
            'guesses': solution['guesses'],
            'piece_shapes': solution['piece_shapes'],
            'target': solution['target'],
            'source': solution['source'],
            'surfaces': solution['surfaces'],        
            'initial_placements': solution['initial_placements'],
            'best_score': solution['best_score'],
            'best_guess': solution.get('best_guess'),
            'best_guess_index': solution.get('best_guess_index', 0),
            'renderer': solution['renderer'],
            'puzzle_pieces': solution['puzzle_pieces'],
            'movement_data': movement_data  # ADD: Pre-calculated movement data
        }
        
        self.logger.info(f"  → Passing {len(solution['guesses'])} guesses to visualizer")
        if movement_data:
            num_movements = len(movement_data.get('movements', {}))
            self.logger.info(f"  → Calculated movement data for {num_movements} pieces")
        
        app = SolverVisualizerApp(solver_data)
        app.run()

