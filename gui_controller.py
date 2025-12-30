#!/usr/bin/env python3
"""
PREN Puzzle Solver - GUI Controller Application
Main control interface for batch puzzle generation and solving
"""

import sys
import time
import os
import threading
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
import numpy as np
import cv2

from src.core.pipeline import PuzzlePipeline, PipelineResult
from src.core.config import Config
from src.utils.logger import setup_logger
from src.vision.mock_puzzle_creator import MockPuzzleGenerator


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
    """Custom widget for displaying puzzle thumbnails"""
    
    def __init__(self, puzzle_data, **kwargs):
        super().__init__(**kwargs)
        self.puzzle_data = puzzle_data
        self.size_hint = (None, None)
        self.size = (dp(120), dp(120))
        self.allow_stretch = True
        self.bind(on_touch_down=self.on_click)
        
        # Load and display thumbnail
        self.load_thumbnail()
        
        # Add border
        with self.canvas.before:
            Color(0.3, 0.3, 0.3, 1)
            self.border = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_border, pos=self._update_border)
    
    def load_thumbnail(self):
        """Load puzzle thumbnail"""
        try:
            if 'thumbnail_path' in self.puzzle_data:
                thumb_path = str(self.puzzle_data['thumbnail_path'])
                print(f"Loading thumbnail from: {thumb_path}")
                if os.path.exists(thumb_path):
                    self.source = thumb_path
                    print(f"  Successfully loaded thumbnail")
                else:
                    print(f"  Thumbnail path doesn't exist: {thumb_path}")
                    self._create_placeholder()
            else:
                print(f"  No thumbnail_path in puzzle_data for puzzle {self.puzzle_data.get('id', '?')}")
                self._create_placeholder()
        except Exception as e:
            print(f"Error loading thumbnail: {e}")
            import traceback
            traceback.print_exc()
            self._create_placeholder()
    
    def _create_placeholder(self):
        """Create a placeholder thumbnail"""
        try:
            placeholder = np.ones((120, 120, 3), dtype=np.uint8) * 200
            cv2.putText(placeholder, f"P{self.puzzle_data.get('id', '?')}", (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            temp_path = f"temp_thumb_{self.puzzle_data.get('id', 0)}.png"
            cv2.imwrite(temp_path, placeholder)
            self.source = temp_path
        except Exception as e:
            print(f"Error creating placeholder: {e}")
    
    def _update_border(self, instance, value):
        self.border.pos = self.pos
        self.border.size = self.size
    
    def on_click(self, instance, touch):
        if self.collide_point(*touch.pos):
            if hasattr(self.parent, 'on_puzzle_selected'):
                self.parent.on_puzzle_selected(self.puzzle_data)
            return True
        return False


class PuzzleGrid(GridLayout):
    """Grid for displaying puzzle thumbnails"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 5
        self.spacing = dp(10)
        self.padding = dp(10)
        self.size_hint_y = None
        self.height = dp(140)  # Start with height for one row
        self.bind(minimum_height=self._update_height)
        
        self.puzzles = []
        self.selected_puzzle = None
    
    def add_puzzle(self, puzzle_data):
        """Add a puzzle to the grid"""
        try:
            print(f"Adding puzzle {puzzle_data.get('id', '?')} to grid")
            thumbnail = PuzzleThumbnail(puzzle_data)
            self.add_widget(thumbnail)
            self.puzzles.append(puzzle_data)
            
            # Update grid height (spacing can be a list [x, y] or a single value)
            rows = (len(self.puzzles) + self.cols - 1) // self.cols
            spacing_value = self.spacing[1] if isinstance(self.spacing, (list, tuple)) else self.spacing
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
        self.height = dp(140)
    
    def on_puzzle_selected(self, puzzle_data):
        """Handle puzzle selection"""
        self.selected_puzzle = puzzle_data
        if hasattr(self.parent, 'on_puzzle_selected'):
            self.parent.on_puzzle_selected(puzzle_data)
    
    def _update_height(self, instance, value):
        # Auto-height based on content
        pass


class ControllerGUI(BoxLayout):
    """Main GUI controller application"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
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
        
        # Title
        title = Label(
            text="PREN Puzzle Solver - Controller",
            size_hint_y=None,
            height=dp(40),
            font_size='24sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        self.add_widget(title)
        
        # Control Panel
        control_panel = self.create_control_panel()
        self.add_widget(control_panel)
        
        # Puzzle Display Area
        self.puzzle_display = self.create_puzzle_display()
        self.add_widget(self.puzzle_display)
        
        # Status and Progress
        status_panel = self.create_status_panel()
        self.add_widget(status_panel)
    
    def create_control_panel(self):
        """Create the main control panel"""
        panel = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(80),
            spacing=dp(15)
        )
        
        # Left: Generation Controls
        gen_controls = BoxLayout(orientation='vertical', spacing=dp(5))
        gen_title = Label(text="Puzzle Generation", font_size='16sp', bold=True, color=(0.3, 0.3, 0.3, 1))
        
        gen_buttons = BoxLayout(orientation='horizontal', spacing=dp(10))
        self.gen_10_btn = Button(text="Generate 10", size_hint_x=0.5)
        self.gen_10_btn.bind(on_press=self.generate_10_puzzles)
        
        self.add_10_btn = Button(text="Add 10 More", size_hint_x=0.5)
        self.add_10_btn.bind(on_press=self.add_10_puzzles)
        
        gen_buttons.add_widget(self.gen_10_btn)
        gen_buttons.add_widget(self.add_10_btn)
        
        gen_controls.add_widget(gen_title)
        gen_controls.add_widget(gen_buttons)
        
        # Middle: Solving Controls
        solve_controls = BoxLayout(orientation='vertical', spacing=dp(5))
        solve_title = Label(text="Batch Solving", font_size='16sp', bold=True, color=(0.3, 0.3, 0.3, 1))
        
        solve_buttons = BoxLayout(orientation='horizontal', spacing=dp(10))
        self.solve_all_btn = Button(text="Solve All", size_hint_x=0.5)
        self.solve_all_btn.bind(on_press=self.solve_all_puzzles)
        
        self.visualize_btn = Button(text="Visualize", size_hint_x=0.5, disabled=True)
        self.visualize_btn.bind(on_press=self.visualize_solution)
        
        solve_buttons.add_widget(self.solve_all_btn)
        solve_buttons.add_widget(self.visualize_btn)
        
        solve_controls.add_widget(solve_title)
        solve_controls.add_widget(solve_buttons)
        
        # Right: Statistics
        stats_controls = BoxLayout(orientation='vertical', spacing=dp(5))
        stats_title = Label(text="Statistics", font_size='16sp', bold=True, color=(0.3, 0.3, 0.3, 1))
        
        self.stats_label = Label(
            text="No puzzles generated",
            size_hint_y=None,
            height=dp(30),
            color=(0.4, 0.4, 0.4, 1)
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
        container = BoxLayout(orientation='vertical', size_hint_y=0.4)
        
        # Title
        title = Label(
            text="Generated Puzzles",
            size_hint_y=None,
            height=dp(30),
            font_size='18sp',
            bold=True,
            color=(0.3, 0.3, 0.3, 1)
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
            orientation='vertical',
            size_hint_y=None,
            height=dp(100),
            spacing=dp(10)
        )
        
        # Status label
        self.status_label = Label(
            text="Ready to generate puzzles",
            size_hint_y=None,
            height=dp(30),
            color=(0.3, 0.3, 0.3, 1),
            font_size='14sp'
        )
        panel.add_widget(self.status_label)
        
        # Progress bar
        self.progress_bar = ProgressBar(
            size_hint_y=None,
            height=dp(20),
            max=100,
            value=0
        )
        panel.add_widget(self.progress_bar)
        
        # Progress label
        self.progress_label = Label(
            text="",
            size_hint_y=None,
            height=dp(20),
            color=(0.5, 0.5, 0.5, 1),
            font_size='12sp'
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
        """Add 10 more puzzles to existing set"""
        if self.current_puzzle_set is None:
            self.status_label.text = "Generate puzzles first!"
            return
        
        self.status_label.text = "Adding 10 more puzzles..."
        self.progress_bar.value = 0
        
        # Generate in background thread
        threading.Thread(target=self._generate_puzzles_thread, args=(10, False)).start()
    
    def _generate_puzzles_thread(self, count: int, clear_existing: bool):
        """Background thread for puzzle generation"""
        try:
            generated_puzzles = []
            
            for i in range(count):
                # Update progress (capture i in lambda)
                progress = (i + 1) / count * 100
                Clock.schedule_once(lambda dt, p=progress: self._update_progress(p))
                
                # Generate puzzle
                puzzle_data = self._generate_single_puzzle(i)
                generated_puzzles.append(puzzle_data)
                
                # Add to UI
                Clock.schedule_once(lambda dt, pd=puzzle_data: self.puzzle_grid.add_puzzle(pd))
                
                # Small delay for visual feedback
                time.sleep(0.1)
            
            # Update puzzle set
            if clear_existing or self.current_puzzle_set is None:
                self.current_puzzle_set = PuzzleSet(
                    id=f"set_{int(time.time())}",
                    timestamp=datetime.now(),
                    puzzles=generated_puzzles
                )
            else:
                self.current_puzzle_set.puzzles.extend(generated_puzzles)
            
            # Update UI
            print(f"Generation complete! Created {len(generated_puzzles)} puzzles")
            Clock.schedule_once(self._puzzle_generation_complete)
            print("Scheduled completion callback")
            
        except Exception as e:
            self.logger.exception(f"Puzzle generation failed: {e}")
            error_msg = str(e)
            print(f"ERROR in generation thread: {error_msg}")
            Clock.schedule_once(lambda dt, msg=error_msg: self._update_status(f"Error: {msg}"))
    
    def _generate_single_puzzle(self, index: int) -> Dict:
        """Generate a single puzzle and save it to its own directory (NO analysis)"""
        # Create output directory for this puzzle
        puzzle_dir = Path(f"data/generated_puzzles/puzzle_{index}")
        puzzle_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save puzzle (no analysis happens here)
        generator = MockPuzzleGenerator(output_dir=str(puzzle_dir))
        full_image, piece_images, debug_image, puzzle_pieces = generator.generate_puzzle_with_positions()
        
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
            'id': index,
            'directory': str(puzzle_dir),
            'thumbnail_path': str(thumbnail_path),
            'debug_path': str(debug_path),
            'piece_paths': list(piece_paths),
            'piece_count': len(piece_paths)
        }
    
    def solve_all_puzzles(self, instance):
        """Solve all generated puzzles"""
        if self.current_puzzle_set is None or len(self.current_puzzle_set.puzzles) == 0:
            self.status_label.text = "Generate puzzles first!"
            return
        
        if self.is_solving:
            self.status_label.text = "Already solving..."
            return
        
        self.status_label.text = f"Solving {len(self.current_puzzle_set.puzzles)} puzzles..."
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
            
            for i, puzzle_data in enumerate(self.current_puzzle_set.puzzles):
                # Update progress
                progress = (i + 1) / total_puzzles * 100
                Clock.schedule_once(lambda dt, p=progress: self._update_progress(p))
                Clock.schedule_once(lambda dt, idx=i, total=total_puzzles: 
                    self._update_status(f"Solving puzzle {idx + 1}/{total}..."))
                
                # Solve puzzle (without UI)
                result = self._solve_single_puzzle(puzzle_data)
                results.append(result)
                
                # Store steps data for visualization
                puzzle_data['solve_result'] = result
                puzzle_data['solved'] = result.success
            
            self.solve_results = results
            self.current_puzzle_set.results = results
            self.current_puzzle_set.solved = True
            
            # Update UI
            Clock.schedule_once(self._solving_complete)
            
        except Exception as e:
            self.logger.exception(f"Solving failed: {e}")
            Clock.schedule_once(lambda dt: self._update_status(f"Solving error: {e}"))
        finally:
            self.is_solving = False
            Clock.schedule_once(lambda dt: setattr(self.solve_all_btn, 'disabled', False))
    
    def _solve_single_puzzle(self, puzzle_data: Dict) -> SolveResult:
        """Solve a single puzzle without UI"""
        start_time = time.time()
        
        try:
            # Create pipeline with the specific puzzle directory
            # This will load the saved pieces and analyze them
            pipeline = PuzzlePipeline(
                self.config, 
                show_ui=False,
                puzzle_dir=puzzle_data['directory']
            )
            result = pipeline.run()
            
            duration = time.time() - start_time
            
            if result.success and result.solution:
                score = result.solution.get('score', 0.0)
                num_guesses = len(result.solution.get('guesses', []))
                
                # Calculate confidence
                max_possible_score = 90000
                confidence = min(100, max(0, (score + 10000) / max_possible_score * 100))
                
                # Store steps data for visualization
                steps_data = {
                    'solution': result.solution,
                    'guesses': result.solution.get('guesses', []),
                    'piece_shapes': result.solution.get('piece_shapes', {}),
                    'target': result.solution.get('target'),
                    'source': result.solution.get('source'),
                    'surfaces': result.solution.get('surfaces', {}),
                    'renderer': result.solution.get('renderer'),
                    'puzzle_pieces': result.solution.get('puzzle_pieces', [])
                }
                
                return SolveResult(
                    puzzle_id=str(puzzle_data['id']),
                    success=True,
                    duration=duration,
                    score=score,
                    num_guesses=num_guesses,
                    confidence=confidence,
                    steps_data=steps_data
                )
            else:
                return SolveResult(
                    puzzle_id=str(puzzle_data['id']),
                    success=False,
                    duration=duration,
                    score=0.0,
                    num_guesses=0,
                    confidence=0.0,
                    error_message=result.message
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Failed to solve puzzle {puzzle_data['id']}: {e}")
            
            return SolveResult(
                puzzle_id=str(puzzle_data['id']),
                success=False,
                duration=duration,
                score=0.0,
                num_guesses=0,
                confidence=0.0,
                    error_message=str(e)
                )
    
    def visualize_solution(self, instance):
        """Visualize the solution for selected puzzle"""
        if self.current_puzzle_set is None:
            self.status_label.text = "No puzzles to visualize!"
            return
        
        # Find first solved puzzle
        solved_puzzle = None
        for puzzle_data in self.current_puzzle_set.puzzles:
            if puzzle_data.get('solved', False) and 'solve_result' in puzzle_data:
                solved_puzzle = puzzle_data
                break
        
        if solved_puzzle is None:
            self.status_label.text = "No solved puzzles to visualize!"
            return
        
        # Launch visualizer with stored data
        try:
            from src.ui.simulator.solver_visualizer import SolverVisualizerApp
            
            steps_data = solved_puzzle['solve_result'].steps_data
            if steps_data and steps_data['solution']:
                app = SolverVisualizerApp(steps_data)
                app.run()
            else:
                self.status_label.text = "No visualization data available!"
                
        except Exception as e:
            self.logger.exception(f"Visualization failed: {e}")
            self.status_label.text = f"Visualization error: {e}"
    
    def on_puzzle_selected(self, puzzle_data):
        """Handle puzzle selection from grid"""
        self.status_label.text = f"Selected puzzle {puzzle_data['id']}"
        
        # Enable visualize button if this puzzle is solved
        if puzzle_data.get('solved', False):
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
        print("_puzzle_generation_complete called!")
        try:
            print(f"current_puzzle_set: {self.current_puzzle_set}")
            if self.current_puzzle_set is None:
                print("ERROR: current_puzzle_set is None!")
                self.status_label.text = "Error: No puzzle set created"
                return
                
            puzzle_count = len(self.current_puzzle_set.puzzles)
            print(f"Setting status text to: Generated {puzzle_count} puzzles successfully")
            self.status_label.text = f"Generated {puzzle_count} puzzles successfully"
            self.progress_bar.value = 100
            print("Calling _update_statistics...")
            self._update_statistics()
            print("_puzzle_generation_complete finished successfully!")
        except Exception as e:
            print(f"EXCEPTION in _puzzle_generation_complete: {e}")
            self.logger.exception(f"Error in _puzzle_generation_complete: {e}")
            self.status_label.text = f"Error completing generation: {e}"
    
    def _solving_complete(self, dt):
        """Called when solving is complete"""
        successful = sum(1 for r in self.solve_results if r.success)
        total = len(self.solve_results)
        avg_score = statistics.mean([r.score for r in self.solve_results if r.success]) if successful > 0 else 0
        
        self.status_label.text = f"Solved {successful}/{total} puzzles (avg score: {avg_score:.0f})"
        self.progress_bar.value = 100
        self.visualize_btn.disabled = False
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
                    avg_duration = statistics.mean([r.duration for r in self.solve_results if r.success])
                    avg_score = statistics.mean([r.score for r in self.solve_results if r.success])
                    avg_confidence = statistics.mean([r.confidence for r in self.solve_results if r.success])
                    
                    self.stats_label.text = (
                        f"Puzzles: {puzzle_count}\n"
                        f"Success: {successful}/{len(self.solve_results)} ({success_rate:.1f}%)\n"
                        f"Avg: {avg_duration:.1f}s, {avg_score:.0f}pts, {avg_confidence:.1f}%"
                    )
                else:
                    self.stats_label.text = f"Puzzles: {puzzle_count}\nFailed: {len(self.solve_results)}"
            else:
                print(f"  No solve results, setting: Puzzles: {puzzle_count}, Not solved yet")
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
            puzzle_dirs = sorted([d for d in puzzles_dir.iterdir() if d.is_dir() and d.name.startswith("puzzle_")])
            
            if not puzzle_dirs:
                print("  No existing puzzles found")
                return
            
            print(f"  Found {len(puzzle_dirs)} existing puzzles")
            
            # Load puzzles
            loaded_puzzles = []
            for puzzle_dir in puzzle_dirs:
                try:
                    # Extract puzzle ID from directory name
                    puzzle_id = int(puzzle_dir.name.split('_')[1])
                    
                    # Check for required files
                    thumbnail_path = puzzle_dir / "thumbnail.png"
                    debug_path = puzzle_dir / "debug_cuts.png"
                    piece_paths = sorted(puzzle_dir.glob("piece_*.png"))
                    
                    if not thumbnail_path.exists() or len(piece_paths) == 0:
                        print(f"  Skipping {puzzle_dir.name} - missing files")
                        continue
                    
                    puzzle_data = {
                        'id': puzzle_id,
                        'directory': str(puzzle_dir),
                        'thumbnail_path': str(thumbnail_path),
                        'debug_path': str(debug_path) if debug_path.exists() else None,
                        'piece_paths': [str(p) for p in piece_paths],
                        'piece_count': len(piece_paths)
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
                    puzzles=loaded_puzzles
                )
                self.status_label.text = f"Loaded {len(loaded_puzzles)} existing puzzles"
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