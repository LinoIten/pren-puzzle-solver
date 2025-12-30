#!/usr/bin/env python3
"""
PREN Puzzle Solver - GUI Controller Application
Main control interface for batch puzzle generation and solving
"""

import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView

from src.core.config import Config
from src.core.pipeline import PuzzlePipeline
from src.solver.validation.scorer import PlacementScorer
from src.ui.simulator.solver_visualizer import SolverVisualizer
from src.utils.logger import setup_logger
from src.vision.mock_puzzle_creator import MockPuzzleGenerator

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class PuzzleSet:
    """Represents a set of generated puzzles"""

    id: str
    timestamp: datetime
    puzzles: List[Dict]  # List of puzzle data
    solved: bool = False
    results: Optional[List] = None


@dataclass
class SolveResult:
    """Result of solving a single puzzle"""

    puzzle_id: str
    success: bool
    duration: float
    score: float
    num_guesses: int
    confidence: float
    error_message: str = ""
    steps_data: Optional[Dict] = None


class PuzzleThumbnail(Image):
    """Custom widget for displaying puzzle thumbnails with selection state"""

    def __init__(self, puzzle_data, **kwargs):
        super().__init__(**kwargs)
        self.puzzle_data = puzzle_data
        self.size_hint = (None, None)
        self.size = (dp(120), dp(120))
        self.allow_stretch = True
        self.keep_ratio = False
        self.bind(on_touch_down=self.on_click)
        self.is_selected = False

        # Load and display thumbnail
        self.load_thumbnail()

        # Add border (will be updated based on selection)
        with self.canvas.before:
            self.border_color = Color(0.3, 0.3, 0.3, 1)  # Store reference
            self.border = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_border, pos=self._update_border)

    def load_thumbnail(self):
        """Load or create appropriate thumbnail based on solve status"""
        try:
            # Check if puzzle is solved
            is_solved = self.puzzle_data.get("solved", False)

            if is_solved and "solve_result" in self.puzzle_data:
                # Create solved thumbnail (best solution visualization)
                self._create_solved_thumbnail()
            else:
                # Create unsolved thumbnail (source pieces layout)
                self._create_unsolved_thumbnail()

        except Exception as e:
            print(
                f"Error creating thumbnail for puzzle {self.puzzle_data.get('id', '?')}: {e}"
            )
            self._create_placeholder()

    def _create_solved_thumbnail(self):
        """Create thumbnail showing the best solution"""
        try:
            steps_data = self.puzzle_data["solve_result"].steps_data
            if not steps_data or not steps_data.get("best_guess"):
                self._create_placeholder()
                return

            # Use the renderer to create the best solution image
            renderer = steps_data.get("renderer")
            best_guess = steps_data.get("best_guess")
            piece_shapes = steps_data.get("piece_shapes", {})

            if renderer and best_guess and piece_shapes:
                # Render the best solution
                rendered_color = renderer.render_debug(best_guess, piece_shapes)

                # Resize to thumbnail size
                thumbnail = cv2.resize(rendered_color, (120, 120))

                # Add "SOLVED" indicator
                cv2.putText(
                    thumbnail,
                    "SOLVED",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    thumbnail,
                    f"#{self.puzzle_data.get('id', '?')}",
                    (5, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # Save and load
                temp_path = f"temp/solved_thumb_{self.puzzle_data.get('id', 0)}.png"
                cv2.imwrite(temp_path, thumbnail)
                self.source = temp_path
            else:
                self._create_placeholder()

        except Exception as e:
            print(f"Error creating solved thumbnail: {e}")
            self._create_placeholder()

    def _create_unsolved_thumbnail(self):
        """Create thumbnail showing source pieces layout (like left side of visualizer)"""
        try:
            # Check if we have the debug image saved
            debug_path = self.puzzle_data.get("debug_path")
            if debug_path and os.path.exists(debug_path):
                # Use the debug image which shows the cuts
                debug_img = cv2.imread(debug_path)
                if debug_img is not None:
                    # Resize to thumbnail size
                    thumbnail = cv2.resize(debug_img, (120, 120))

                    # Add "UNSOLVED" indicator
                    cv2.putText(
                        thumbnail,
                        "UNSOLVED",
                        (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1,
                    )
                    cv2.putText(
                        thumbnail,
                        f"#{self.puzzle_data.get('id', '?')}",
                        (5, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                    # Save and load
                    temp_path = (
                        f"temp/unsolved_thumb_{self.puzzle_data.get('id', 0)}.png"
                    )
                    cv2.imwrite(temp_path, thumbnail)
                    self.source = temp_path
                    return

            # Fallback: create from piece images if available
            piece_paths = self.puzzle_data.get("piece_paths", [])
            if piece_paths:
                # Create a simple grid layout of pieces
                thumbnail = np.ones((120, 120, 3), dtype=np.uint8) * 240

                # Load and arrange first few pieces in a grid
                pieces_per_row = 2
                piece_size = 50
                start_x, start_y = 10, 10

                for i, piece_path in enumerate(piece_paths[:4]):  # Show max 4 pieces
                    if os.path.exists(piece_path):
                        piece_img = cv2.imread(piece_path)
                        if piece_img is not None:
                            piece_small = cv2.resize(
                                piece_img, (piece_size, piece_size)
                            )

                            row = i // pieces_per_row
                            col = i % pieces_per_row
                            y = start_y + row * (piece_size + 5)
                            x = start_x + col * (piece_size + 5)

                            if y + piece_size <= 120 and x + piece_size <= 120:
                                thumbnail[y : y + piece_size, x : x + piece_size] = (
                                    piece_small
                                )

                # Add labels
                cv2.putText(
                    thumbnail,
                    "UNSOLVED",
                    (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    thumbnail,
                    f"#{self.puzzle_data.get('id', '?')}",
                    (5, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

                # Save and load
                temp_path = f"temp/unsolved_thumb_{self.puzzle_data.get('id', 0)}.png"
                cv2.imwrite(temp_path, thumbnail)
                self.source = temp_path
            else:
                self._create_placeholder()

        except Exception as e:
            print(f"Error creating unsolved thumbnail: {e}")
            self._create_placeholder()

    def _create_placeholder(self):
        """Create a placeholder thumbnail"""
        try:
            placeholder = np.ones((120, 120, 3), dtype=np.uint8) * 200
            cv2.putText(
                placeholder,
                f"P{self.puzzle_data.get('id', '?')}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
            temp_path = f"temp/thumb_{self.puzzle_data.get('id', 0)}.png"
            cv2.imwrite(temp_path, placeholder)
            self.source = temp_path
        except Exception as e:
            print(f"Error creating placeholder: {e}")

    def set_selected(self, selected):
        """Update selection state and border color"""
        self.is_selected = selected
        if selected:
            self.border_color.rgba = (0, 0.8, 0, 1)  # Green for selected
        else:
            # Color based on solve status
            if self.puzzle_data.get("solved", False):
                self.border_color.rgba = (0, 0.6, 0, 1)  # Dark green for solved
            else:
                self.border_color.rgba = (0.6, 0.6, 0.6, 1)  # Gray for unsolved

    def refresh_thumbnail(self):
        """Refresh the thumbnail (call this after puzzle is solved)"""
        self.load_thumbnail()
        self.set_selected(self.is_selected)  # Maintain selection state

    def _update_border(self, instance, value):
        self.border.pos = self.pos
        self.border.size = self.size

    def on_click(self, instance, touch):
        if self.collide_point(*touch.pos):
            if hasattr(self.parent, "on_puzzle_selected"):
                self.parent.on_puzzle_selected(self.puzzle_data, self)
            return True
        return False


class PuzzleGrid(GridLayout):
    """Grid for displaying puzzle thumbnails with selection"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 5
        self.spacing = dp(10)
        self.padding = dp(10)
        self.size_hint_y = None
        self.height = dp(140)
        self.bind(minimum_height=self._update_height)  # This was causing the error

        self.puzzles = []
        self.selected_puzzle = None
        self.selected_thumbnail = None

    def add_puzzle(self, puzzle_data):
        """Add a puzzle to the grid"""
        try:
            print(f"Adding puzzle {puzzle_data.get('id', '?')} to grid")
            thumbnail = PuzzleThumbnail(puzzle_data)
            self.add_widget(thumbnail)
            self.puzzles.append(puzzle_data)

            # Update grid height
            rows = (len(self.puzzles) + self.cols - 1) // self.cols
            spacing_value = (
                self.spacing[1]
                if isinstance(self.spacing, (list, tuple))
                else self.spacing
            )
            self.height = rows * dp(140) + (rows - 1) * spacing_value
            print(f"  Successfully added puzzle to grid (total: {len(self.puzzles)})")
        except Exception as e:
            print(f"ERROR adding puzzle to grid: {e}")
            import traceback

            traceback.print_exc()

    def clear_puzzles(self):
        """Clear all puzzles from grid"""
        self.clear_widgets()
        self.puzzles = []
        self.selected_puzzle = None
        self.selected_thumbnail = None
        self.height = dp(140)

    def on_puzzle_selected(self, puzzle_data, thumbnail_widget):
        """Handle puzzle selection"""
        # Clear previous selection
        if self.selected_thumbnail:
            self.selected_thumbnail.set_selected(False)

        # Set new selection
        self.selected_puzzle = puzzle_data
        self.selected_thumbnail = thumbnail_widget
        thumbnail_widget.set_selected(True)

        # Notify parent
        if hasattr(self.parent, "on_puzzle_selected"):
            self.parent.on_puzzle_selected(puzzle_data)

    def refresh_puzzle_thumbnails(self):
        """Refresh all thumbnails (call after solving)"""
        for widget in self.children:
            if isinstance(widget, PuzzleThumbnail):
                widget.refresh_thumbnail()

    def _update_height(self, instance, value):
        """Update height based on content - this method was missing"""
        # This method is called when minimum_height changes
        # We handle height updates manually in add_puzzle, so this can be minimal
        pass


class ControllerGUI(BoxLayout):
    """Main GUI controller application"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(20)
        self.spacing = dp(15)

        # Initialize components
        self.logger = setup_logger("controller_gui")
        self.config = Config()
        self.puzzle_generator = MockPuzzleGenerator(output_dir="data/mock_pieces")
        self.current_puzzle_set = None
        self.solve_results = []
        self.is_solving = False

        # Setup UI
        self.setup_ui()

        # Load existing puzzles if any
        Clock.schedule_once(lambda dt: self.load_existing_puzzles(), 0.5)

        # Background
        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.bg = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_bg, pos=self._update_bg)

    def setup_ui(self):
        """Setup the main UI layout"""
        # Make the main container horizontal to fit viz panel on the side
        self.main_container = BoxLayout(orientation="horizontal", spacing=dp(15))

        # Left side: existing content
        left_panel = BoxLayout(orientation="vertical", size_hint_x=0.6)

        # Title
        title = Label(
            text="PREN Puzzle Solver - Controller",
            size_hint_y=None,
            height=dp(40),
            font_size="24sp",
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
        )
        left_panel.add_widget(title)

        # Control Panel
        control_panel = self.create_control_panel()
        left_panel.add_widget(control_panel)

        # Puzzle Display Area
        puzzle_display = self.create_puzzle_display()
        left_panel.add_widget(puzzle_display)

        # Status and Progress
        status_panel = self.create_status_panel()
        left_panel.add_widget(status_panel)

        self.main_container.add_widget(left_panel)

        # Right side: visualization panel (initially hidden)
        self.viz_panel = self.create_visualization_panel()
        self.viz_panel.size_hint_x = 0  # Start hidden
        self.viz_panel.opacity = 0
        self.main_container.add_widget(self.viz_panel)

        self.add_widget(self.main_container)

    def create_control_panel(self):
        """Create the main control panel"""
        panel = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(80), spacing=dp(15)
        )

        # Left: Generation Controls
        gen_controls = BoxLayout(orientation="vertical", spacing=dp(5))
        gen_title = Label(
            text="Puzzle Generation",
            font_size="16sp",
            bold=True,
            color=(0.3, 0.3, 0.3, 1),
        )

        gen_buttons = BoxLayout(orientation="horizontal", spacing=dp(10))
        self.add_10_btn = Button(text="Add 10 Puzzles")
        self.add_10_btn.bind(on_press=self.add_10_puzzles)

        self.clear_btn = Button(text="Clear All")
        self.clear_btn.bind(on_press=self.clear_all_puzzles)

        gen_buttons.add_widget(self.add_10_btn)
        gen_buttons.add_widget(self.clear_btn)

        gen_controls.add_widget(gen_title)
        gen_controls.add_widget(gen_buttons)

        solve_controls = BoxLayout(orientation="vertical", spacing=dp(5))
        solve_title = Label(
            text="Batch Solving", font_size="16sp", bold=True, color=(0.3, 0.3, 0.3, 1)
        )

        solve_buttons = BoxLayout(orientation="horizontal", spacing=dp(10))
        self.solve_all_btn = Button(text="Solve All", size_hint_x=0.5)
        self.solve_all_btn.bind(on_press=self.solve_all_puzzles)

        self.visualize_btn = Button(text="Visualize", size_hint_x=0.5, disabled=True)
        self.visualize_btn.bind(on_press=self.visualize_solution)

        solve_buttons.add_widget(self.solve_all_btn)
        solve_buttons.add_widget(self.visualize_btn)

        solve_controls.add_widget(solve_title)
        solve_controls.add_widget(solve_buttons)

        # Right: Statistics
        stats_controls = BoxLayout(orientation="vertical", spacing=dp(5))
        stats_title = Label(
            text="Statistics", font_size="16sp", bold=True, color=(0.3, 0.3, 0.3, 1)
        )

        self.stats_label = Label(
            text="No puzzles generated",
            size_hint_y=None,
            height=dp(30),
            color=(0.4, 0.4, 0.4, 1),
        )

        stats_controls.add_widget(stats_title)
        stats_controls.add_widget(self.stats_label)

        # Add all panels
        panel.add_widget(gen_controls)
        panel.add_widget(solve_controls)
        panel.add_widget(stats_controls)

        return panel

    def create_puzzle_display(self):
        """Create the puzzle display area"""
        # Container for puzzle grid
        container = BoxLayout(orientation="vertical", size_hint_y=0.4)

        # Title
        title = Label(
            text="Generated Puzzles",
            size_hint_y=None,
            height=dp(30),
            font_size="18sp",
            bold=True,
            color=(0.3, 0.3, 0.3, 1),
        )
        container.add_widget(title)

        # Scrollable grid
        scroll = ScrollView()
        self.puzzle_grid = PuzzleGrid()
        scroll.add_widget(self.puzzle_grid)
        container.add_widget(scroll)

        return container

    def create_status_panel(self):
        """Create the status and progress panel"""
        panel = BoxLayout(
            orientation="vertical", size_hint_y=None, height=dp(100), spacing=dp(10)
        )

        # Status label
        self.status_label = Label(
            text="Ready to generate puzzles",
            size_hint_y=None,
            height=dp(30),
            color=(0.3, 0.3, 0.3, 1),
            font_size="14sp",
        )
        panel.add_widget(self.status_label)

        # Progress bar
        self.progress_bar = ProgressBar(
            size_hint_y=None, height=dp(20), max=100, value=0
        )
        panel.add_widget(self.progress_bar)

        # Progress label
        self.progress_label = Label(
            text="",
            size_hint_y=None,
            height=dp(20),
            color=(0.5, 0.5, 0.5, 1),
            font_size="12sp",
        )
        panel.add_widget(self.progress_label)

        return panel

    def generate_10_puzzles(self, instance):
        """Generate 10 new puzzles"""
        self.status_label.text = "Generating 10 new puzzles..."
        self.progress_bar.value = 0

        # Clear existing puzzles
        self.puzzle_grid.clear_puzzles()
        self.current_puzzle_set = None
        self.solve_results = []

        # Generate in background thread
        threading.Thread(target=self._generate_puzzles_thread, args=(10, True)).start()

    def add_10_puzzles(self, instance):
        """Add 10 puzzles to the current set"""
        current_count = (
            len(self.current_puzzle_set.puzzles) if self.current_puzzle_set else 0
        )
        new_count = current_count + 10

        self.status_label.text = (
            f"Generating puzzles {current_count + 1}-{new_count}..."
        )
        self.progress_bar.value = 0

        # Generate in background thread
        threading.Thread(
            target=self._generate_puzzles_thread, args=(10, current_count)
        ).start()

    def clear_all_puzzles(self, instance):
        """Clear all puzzles and reset everything"""
        self.status_label.text = "Clearing all puzzles..."

        # Clear UI
        self.puzzle_grid.clear_puzzles()
        self.current_puzzle_set = None
        self.solve_results = []

        # Hide visualization if open
        if hasattr(self, "viz_panel") and self.viz_panel.opacity > 0:
            self.hide_visualization()

        # Clean up files
        self._cleanup_existing_puzzles()

        self.status_label.text = "Ready to generate puzzles"
        self.progress_bar.value = 0
        self._update_statistics()

    def _generate_puzzles_thread(self, count: int, start_index: int):
        """Background thread for puzzle generation - always generates fresh puzzles"""
        try:
            generated_puzzles = []

            for i in range(count):
                puzzle_index = start_index + i

                # Update progress
                progress = (i + 1) / count * 100
                Clock.schedule_once(lambda dt, p=progress: self._update_progress(p))

                # Generate puzzle with correct index - this creates fresh puzzle files
                puzzle_data = self._generate_single_puzzle(puzzle_index)
                generated_puzzles.append(puzzle_data)

                # Add to UI
                Clock.schedule_once(
                    lambda dt, pd=puzzle_data: self.puzzle_grid.add_puzzle(pd)
                )

                time.sleep(0.1)

            # Update puzzle set
            if start_index == 0 or self.current_puzzle_set is None:
                # Starting fresh
                self.current_puzzle_set = PuzzleSet(
                    id=f"set_{int(time.time())}",
                    timestamp=datetime.now(),
                    puzzles=generated_puzzles,
                )
            else:
                # Adding to existing set
                self.current_puzzle_set.puzzles.extend(generated_puzzles)

            Clock.schedule_once(self._puzzle_generation_complete)

        except Exception as e:
            self.logger.exception(f"Puzzle generation failed: {e}")
            error_msg = str(e)
            Clock.schedule_once(
                lambda dt, msg=error_msg: self._update_status(f"Error: {msg}")
            )

    def _generate_single_puzzle(self, index: int) -> Dict:
        """Generate a single puzzle and save it to its own directory (NO analysis)"""
        # Create output directory for this puzzle
        puzzle_dir = Path(f"data/generated_puzzles/puzzle_{index}")
        puzzle_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save puzzle (no analysis happens here)
        generator = MockPuzzleGenerator(output_dir=str(puzzle_dir))
        full_image, piece_images, debug_image, puzzle_pieces = (
            generator.generate_puzzle_with_positions()
        )

        # Save debug image
        debug_path = puzzle_dir / "debug_cuts.png"
        cv2.imwrite(str(debug_path), debug_image)

        # Create thumbnail from full image
        thumbnail_path = puzzle_dir / "thumbnail.png"
        thumbnail = cv2.resize(full_image, (120, 120))
        cv2.imwrite(str(thumbnail_path), thumbnail)

        # Get saved piece paths
        piece_paths = sorted(puzzle_dir.glob("piece_*.png"))

        return {
            "id": index,
            "directory": str(puzzle_dir),
            "thumbnail_path": str(thumbnail_path),
            "debug_path": str(debug_path),
            "piece_paths": list(piece_paths),
            "piece_count": len(piece_paths),
        }

    def solve_all_puzzles(self, instance):
        """Solve all generated puzzles"""
        if self.current_puzzle_set is None or len(self.current_puzzle_set.puzzles) == 0:
            self.status_label.text = "Generate puzzles first!"
            return

        if self.is_solving:
            self.status_label.text = "Already solving..."
            return

        self.status_label.text = (
            f"Solving {len(self.current_puzzle_set.puzzles)} puzzles..."
        )
        self.progress_bar.value = 0
        self.is_solving = True
        self.solve_all_btn.disabled = True

        # Solve in background thread
        threading.Thread(target=self._solve_all_puzzles_thread).start()

    def _solve_all_puzzles_thread(self):
        """Background thread for solving all puzzles"""
        try:
            results = []
            total_puzzles = len(self.current_puzzle_set.puzzles)

            # Start overall timing
            overall_start_time = time.time()

            for i, puzzle_data in enumerate(self.current_puzzle_set.puzzles):
                # Update progress
                progress = (i + 1) / total_puzzles * 100
                Clock.schedule_once(lambda dt, p=progress: self._update_progress(p))
                Clock.schedule_once(
                    lambda dt, idx=i, total=total_puzzles: self._update_status(
                        f"Solving puzzle {idx + 1}/{total}..."
                    )
                )

                # Solve puzzle (without UI) - use individual timing here
                result = self._solve_single_puzzle(puzzle_data)
                results.append(result)

                # Store steps data for visualization
                puzzle_data["solve_result"] = result
                puzzle_data["solved"] = result.success

            # Calculate total elapsed time
            total_elapsed = time.time() - overall_start_time

            self.solve_results = results
            self.current_puzzle_set.results = results
            self.current_puzzle_set.solved = True

            # Store the accurate total time
            self.total_solve_time = total_elapsed

            # Update UI
            Clock.schedule_once(self._solving_complete)

        except Exception as e:
            self.logger.exception(f"Solving failed: {e}")
            Clock.schedule_once(lambda dt: self._update_status(f"Solving error: {e}"))
        finally:
            self.is_solving = False
            Clock.schedule_once(
                lambda dt: setattr(self.solve_all_btn, "disabled", False)
            )

    def _solve_single_puzzle(self, puzzle_data: Dict) -> SolveResult:
        """Solve a single puzzle without UI"""
        start_time = time.time()

        try:
            # Create pipeline with the specific puzzle directory
            pipeline = PuzzlePipeline(
                self.config, show_ui=False, puzzle_dir=puzzle_data["directory"]
            )
            result = pipeline.run()

            duration = time.time() - start_time

            if result.success and result.solution:
                score = result.solution.get("score", 0.0)
                num_guesses = len(result.solution.get("guesses", []))

                # Calculate confidence
                max_possible_score = 90000
                confidence = min(
                    100, max(0, (score + 10000) / max_possible_score * 100)
                )

                # Calculate best score for visualization
                guesses = result.solution.get("guesses", [])
                best_score = 0.0
                best_guess = None
                best_guess_index = 0

                if (
                    guesses
                    and "renderer" in result.solution
                    and "piece_shapes" in result.solution
                    and "target" in result.solution
                ):
                    try:
                        renderer = result.solution["renderer"]
                        piece_shapes = result.solution["piece_shapes"]
                        target = result.solution["target"]
                        scorer = PlacementScorer(
                            overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5
                        )

                        for i, guess in enumerate(guesses):
                            try:
                                rendered = renderer.render(guess, piece_shapes)
                                guess_score = scorer.score(rendered, target)
                                if guess_score > best_score:
                                    best_score = guess_score
                                    best_guess = guess
                                    best_guess_index = i
                            except:
                                continue

                    except Exception as e:
                        print(f"Error calculating best score: {e}")
                        best_score = score

                # Store steps data for visualization with best_score included
                steps_data = {
                    "solution": result.solution,
                    "guesses": result.solution.get("guesses", []),
                    "piece_shapes": result.solution.get("piece_shapes", {}),
                    "target": result.solution.get("target"),
                    "source": result.solution.get("source"),
                    "surfaces": result.solution.get("surfaces", {}),
                    "renderer": result.solution.get("renderer"),
                    "puzzle_pieces": result.solution.get("puzzle_pieces", []),
                    "best_score": best_score,  # Add this line
                    "best_guess": best_guess,
                    "best_guess_index": best_guess_index,
                    "movement_data": result.solution.get("movement_data", {}),
                }

                return SolveResult(
                    puzzle_id=str(puzzle_data["id"]),
                    success=True,
                    duration=duration,
                    score=score,
                    num_guesses=num_guesses,
                    confidence=confidence,
                    steps_data=steps_data,
                )
            else:
                return SolveResult(
                    puzzle_id=str(puzzle_data["id"]),
                    success=False,
                    duration=duration,
                    score=0.0,
                    num_guesses=0,
                    confidence=0.0,
                    error_message=result.message,
                )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Failed to solve puzzle {puzzle_data['id']}: {e}")

            return SolveResult(
                puzzle_id=str(puzzle_data["id"]),
                success=False,
                duration=duration,
                score=0.0,
                num_guesses=0,
                confidence=0.0,
                error_message=str(e),
            )

    def create_visualization_panel(self):
        """Create the visualization panel using the actual SolverVisualizer"""
        panel = BoxLayout(orientation="vertical", spacing=dp(10))

        # Title
        viz_title = Label(
            text="Solution Visualization",
            size_hint_y=None,
            height=dp(30),
            font_size="18sp",
            bold=True,
            color=(0.3, 0.3, 0.3, 1),
        )
        panel.add_widget(viz_title)

        # Placeholder for the actual visualizer (will be added dynamically)
        self.viz_container = BoxLayout(orientation="vertical")
        panel.add_widget(self.viz_container)

        # Close button
        close_btn = Button(
            text="✕ Close Visualization",
            size_hint_y=None,
            height=dp(40),
            font_size="12sp",
        )
        close_btn.bind(on_press=self.hide_visualization)
        panel.add_widget(close_btn)

        return panel

    def show_visualization_panel(self, steps_data):
        """Show the visualization panel using the actual SolverVisualizer"""
        # Clear any existing visualizer
        self.viz_container.clear_widgets()

        # Create the embedded visualizer (steps_data already has best_score)
        self.embedded_visualizer = SolverVisualizer(steps_data, embedded=True)

        # Add it to the container
        self.viz_container.add_widget(self.embedded_visualizer)

        # Show the panel with animation
        from kivy.animation import Animation

        # Adjust left panel size
        anim_left = Animation(size_hint_x=0.55, duration=0.3)
        anim_left.start(self.main_container.children[1])

        # Show viz panel
        self.viz_panel.size_hint_x = 0.45
        anim_viz = Animation(opacity=1, duration=0.3)
        anim_viz.start(self.viz_panel)

    def hide_visualization(self, instance=None):
        """Hide the visualization panel"""
        from kivy.animation import Animation

        # Stop any running visualization
        if hasattr(self, "embedded_visualizer") and self.embedded_visualizer:
            if (
                hasattr(self.embedded_visualizer, "is_running")
                and self.embedded_visualizer.is_running
            ):
                self.embedded_visualizer.pause_visualization(None)

        # Hide viz panel
        anim_viz = Animation(opacity=0, duration=0.3)
        anim_viz.bind(on_complete=self._on_viz_hidden)
        anim_viz.start(self.viz_panel)

        # Restore left panel size
        anim_left = Animation(size_hint_x=1.0, duration=0.3)
        anim_left.start(self.main_container.children[1])

    def _on_viz_hidden(self, animation, widget):
        """Called when visualization panel is fully hidden"""
        self.viz_panel.size_hint_x = 0
        # Clear the visualizer to free memory
        if hasattr(self, "viz_container"):
            self.viz_container.clear_widgets()

    def on_puzzle_selected(self, puzzle_data):
        """Handle puzzle selection from grid"""
        self.status_label.text = f"Selected puzzle {puzzle_data['id']}"

        # Enable visualize button if this puzzle is solved
        if puzzle_data.get("solved", False):
            self.visualize_btn.disabled = False
        else:
            self.visualize_btn.disabled = True

    def _update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.value = value
        self.progress_label.text = f"{value:.0f}%"

    def _update_status(self, text):
        """Update status label"""
        self.status_label.text = text

    def _puzzle_generation_complete(self, dt):
        """Called when puzzle generation is complete"""
        try:
            if self.current_puzzle_set is None:
                self.status_label.text = "Error: No puzzle set created"
                return

            puzzle_count = len(self.current_puzzle_set.puzzles)
            self.status_label.text = f"Total: {puzzle_count} puzzles generated"
            self.progress_bar.value = 100

            self._update_statistics()

        except Exception as e:
            self.logger.exception(f"Error in _puzzle_generation_complete: {e}")
            self.status_label.text = f"Error completing generation: {e}"

    def _solving_complete(self, dt):
        """Called when solving is complete"""
        successful = sum(1 for r in self.solve_results if r.success)
        total = len(self.solve_results)
        avg_score = (
            statistics.mean([r.score for r in self.solve_results if r.success])
            if successful > 0
            else 0
        )

        # Use the accurate total time
        total_time = getattr(self, "total_solve_time", 0)
        avg_time = total_time / total if total > 0 else 0

        self.status_label.text = (
            f"Solved {successful}/{total} puzzles (avg score: {avg_score:.0f}, "
            f"total time: {total_time:.1f}s, avg: {avg_time:.1f}s)"
        )
        self.progress_bar.value = 100
        self.visualize_btn.disabled = False

        # Refresh all thumbnails to show solved state
        self.puzzle_grid.refresh_puzzle_thumbnails()

        self._update_statistics()

    def _update_statistics(self):
        """Update statistics display"""
        print("_update_statistics called")
        try:
            if self.current_puzzle_set is None:
                print("  current_puzzle_set is None, setting default text")
                self.stats_label.text = "No puzzles generated"
                return

            puzzle_count = len(self.current_puzzle_set.puzzles)
            print(f"  puzzle_count: {puzzle_count}")

            if self.solve_results:
                successful = sum(1 for r in self.solve_results if r.success)
                success_rate = successful / len(self.solve_results) * 100

                if successful > 0:
                    # Use the accurate total time instead of individual durations
                    total_time = getattr(self, "total_solve_time", 0)
                    avg_time = (
                        total_time / len(self.solve_results)
                        if len(self.solve_results) > 0
                        else 0
                    )

                    avg_score = statistics.mean(
                        [r.score for r in self.solve_results if r.success]
                    )
                    avg_confidence = statistics.mean(
                        [r.confidence for r in self.solve_results if r.success]
                    )

                    self.stats_label.text = (
                        f"Puzzles: {puzzle_count}\n"
                        f"Success: {successful}/{len(self.solve_results)} ({success_rate:.1f}%)\n"
                        f"Total: {total_time:.1f}s, Avg: {avg_time:.1f}s\n"
                        f"Score: {avg_score:.0f}pts, Conf: {avg_confidence:.1f}%"
                    )
                else:
                    self.stats_label.text = (
                        f"Puzzles: {puzzle_count}\nFailed: {len(self.solve_results)}"
                    )
            else:
                print(
                    f"  No solve results, setting: Puzzles: {puzzle_count}, Not solved yet"
                )
                self.stats_label.text = f"Puzzles: {puzzle_count}\nNot solved yet"
            print("_update_statistics completed successfully")
        except Exception as e:
            print(f"EXCEPTION in _update_statistics: {e}")
            import traceback

            traceback.print_exc()

    def load_existing_puzzles(self):
        """Load existing puzzles from the generated_puzzles directory"""
        try:
            print("Checking for existing puzzles...")
            puzzles_dir = Path("data/generated_puzzles")
            if not puzzles_dir.exists():
                print("  No generated_puzzles directory found")
                return

            # Find all puzzle directories
            puzzle_dirs = sorted(
                [
                    d
                    for d in puzzles_dir.iterdir()
                    if d.is_dir() and d.name.startswith("puzzle_")
                ]
            )

            if not puzzle_dirs:
                print("  No existing puzzles found")
                return

            print(f"  Found {len(puzzle_dirs)} existing puzzles")

            # Load puzzles
            loaded_puzzles = []
            for puzzle_dir in puzzle_dirs:
                try:
                    # Extract puzzle ID from directory name
                    puzzle_id = int(puzzle_dir.name.split("_")[1])

                    # Check for required files
                    thumbnail_path = puzzle_dir / "thumbnail.png"
                    debug_path = puzzle_dir / "debug_cuts.png"
                    piece_paths = sorted(puzzle_dir.glob("piece_*.png"))

                    if not thumbnail_path.exists() or len(piece_paths) == 0:
                        print(f"  Skipping {puzzle_dir.name} - missing files")
                        continue

                    puzzle_data = {
                        "id": puzzle_id,
                        "directory": str(puzzle_dir),
                        "thumbnail_path": str(thumbnail_path),
                        "debug_path": str(debug_path) if debug_path.exists() else None,
                        "piece_paths": [str(p) for p in piece_paths],
                        "piece_count": len(piece_paths),
                    }

                    loaded_puzzles.append(puzzle_data)
                    self.puzzle_grid.add_puzzle(puzzle_data)

                except Exception as e:
                    print(f"  Error loading {puzzle_dir.name}: {e}")
                    continue

            if loaded_puzzles:
                # Create puzzle set
                self.current_puzzle_set = PuzzleSet(
                    id=f"loaded_{int(time.time())}",
                    timestamp=datetime.now(),
                    puzzles=loaded_puzzles,
                )
                self.status_label.text = (
                    f"Loaded {len(loaded_puzzles)} existing puzzles"
                )
                self._update_statistics()
                print(f"  Successfully loaded {len(loaded_puzzles)} puzzles")

        except Exception as e:
            print(f"Error loading existing puzzles: {e}")
            import traceback

            traceback.print_exc()

    def _update_bg(self, instance, value):
        """Update background"""
        self.bg.pos = self.pos
        self.bg.size = self.size

    def visualize_solution(self, instance):
        """Show visualization panel for selected puzzle"""
        if self.current_puzzle_set is None:
            self.status_label.text = "No puzzles to visualize!"
            return

        # Find solved puzzle
        solved_puzzle = None
        if (
            hasattr(self.puzzle_grid, "selected_puzzle")
            and self.puzzle_grid.selected_puzzle
        ):
            if self.puzzle_grid.selected_puzzle.get("solved", False):
                solved_puzzle = self.puzzle_grid.selected_puzzle

        if solved_puzzle is None:
            for puzzle_data in self.current_puzzle_set.puzzles:
                if puzzle_data.get("solved", False) and "solve_result" in puzzle_data:
                    solved_puzzle = puzzle_data
                    break

        if solved_puzzle is None:
            self.status_label.text = "No solved puzzles to visualize!"
            return

        try:
            steps_data = solved_puzzle["solve_result"].steps_data
            if steps_data and steps_data["solution"]:
                # Simply pass the data to the actual visualizer!
                self.show_visualization_panel(steps_data)
            else:
                self.status_label.text = "No visualization data available!"

        except Exception as e:
            self.logger.exception(f"Visualization failed: {e}")
            self.status_label.text = f"Visualization error: {e}"

    def generate_puzzles(self, instance):
        """Generate 10 puzzles - adds to existing set or starts fresh"""
        existing_count = (
            len(self.current_puzzle_set.puzzles) if self.current_puzzle_set else 0
        )

        if existing_count == 0:
            self.status_label.text = "Generating 10 new puzzles..."
            start_index = 0
            # Clean up any leftover files
            self._cleanup_existing_puzzles()
        else:
            self.status_label.text = (
                f"Adding 10 more puzzles (currently have {existing_count})..."
            )
            start_index = existing_count

        self.progress_bar.value = 0

        # Generate in background thread
        threading.Thread(
            target=self._generate_puzzles_thread, args=(10, start_index)
        ).start()

    def _cleanup_existing_puzzles(self):
        """Delete existing generated puzzle directories and temp files"""
        try:
            import shutil

            # Clean up generated puzzles directory
            puzzles_dir = Path("data/generated_puzzles")
            if puzzles_dir.exists():
                print(f"Cleaning up existing puzzles in {puzzles_dir}")
                shutil.rmtree(puzzles_dir)
                print("  Existing puzzles cleaned up")

            # Clean up temporary thumbnail files in current directory
            current_dir = Path(".")
            temp_files = list(current_dir.glob("temp_*thumb_*.png"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    print(f"  Deleted temp file: {temp_file}")
                except Exception as e:
                    print(f"  Warning: Could not delete {temp_file}: {e}")

            if temp_files:
                print(f"Cleaned up {len(temp_files)} temporary thumbnail files")

        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Don't let cleanup errors prevent puzzle generation
            pass


class ControllerApp(App):
    """Main Kivy application"""

    def build(self):
        """Build the application"""
        # Set window properties
        Window.title = "PREN Puzzle Solver - Controller"
        Window.size = (1200, 800)
        Window.minimum_size = (800, 600)

        return ControllerGUI()


def main():
    """Main entry point"""
    ControllerApp().run()


if __name__ == "__main__":
    main()
