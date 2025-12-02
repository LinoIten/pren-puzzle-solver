
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
import numpy as np
import cv2

class SolverVisualizer(BoxLayout):
    def __init__(self, solver_data, **kwargs):
        super().__init__(**kwargs)
        
        # DEBUG: Check what data we have
        print("\n=== SOLVER VISUALIZER DEBUG ===")
        print("Available keys in solver_data:")
        for key in solver_data.keys():
            print(f"  - {key}: {type(solver_data[key])}")
        
        if 'puzzle_pieces' in solver_data:
            pieces = solver_data['puzzle_pieces']
            print(f"\nPuzzle pieces found: {len(pieces)} pieces")
            for piece in pieces[:2]:  # Show first 2
                print(f"  Piece {piece.id}: pick_pose={piece.pick_pose}")
        else:
            print("\n❌ NO puzzle_pieces in solver_data!")
        
        if 'surfaces' in solver_data:
            surfaces = solver_data['surfaces']
            print(f"\nSurfaces found: {surfaces['global']['width']}x{surfaces['global']['height']}")
        else:
            print("\n❌ NO surfaces in solver_data!")
        
        print("===============================\n")
        
        self.orientation = 'vertical'
        self.solver_data = solver_data
        self.current_guess_index = 0
        self.is_running = False
        self.speed = 0.05  # seconds per guess
        
        # Set background color to light grey
        with self.canvas.before:
            Color(0.9, 0.9, 0.9, 1)  # Light grey background
            self.bg = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_bg, pos=self._update_bg)
        
        # Top: Image display
        self.image_widget = Image(size_hint_y=0.75)
        self.add_widget(self.image_widget)
        
        # Bottom: Controls with better styling
        controls = BoxLayout(size_hint_y=0.25, orientation='vertical', padding=20, spacing=15)
        
        # Status label with better styling
        self.status_label = Label(
            text='Ready to visualize', 
            size_hint_y=0.2,
            color=(0.2, 0.2, 0.2, 1),  # Dark grey text
            font_size='16sp',
            bold=True
        )
        controls.add_widget(self.status_label)
        
        # Main controls - First row (Navigation & Playback)
        button_row1 = BoxLayout(orientation='horizontal', size_hint_y=0.4, spacing=10)
        
        # Navigation buttons
        self.first_button = Button(
            text='️First', 
            background_color=(0.3, 0.5, 0.8, 1),  # Blue
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.first_button.bind(on_press=self.go_to_first)
        button_row1.add_widget(self.first_button)
        
        self.back_button = Button(
            text='Back', 
            background_color=(0.7, 0.5, 0.2, 1),  # Orange
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.back_button.bind(on_press=self.go_back)
        button_row1.add_widget(self.back_button)
        
        self.step_button = Button(
            text='Next', 
            background_color=(0.4, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.step_button.bind(on_press=self.step_guess)
        button_row1.add_widget(self.step_button)
        
        # Playback controls
        self.start_button = Button(
            text='Play',
            background_color=(0.3, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.start_button.bind(on_press=self.start_visualization)
        button_row1.add_widget(self.start_button)
        
        self.pause_button = Button(
            text='Pause',
            background_color=(0.8, 0.5, 0.2, 1),  # Orange-red
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.pause_button.bind(on_press=self.pause_visualization)
        button_row1.add_widget(self.pause_button)
        
        self.best_button = Button(
            text='Best',
            background_color=(0.8, 0.2, 0.8, 1),  # Purple
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.best_button.bind(on_press=self.show_best)
        button_row1.add_widget(self.best_button)
        
        controls.add_widget(button_row1)
        
        
        self.add_widget(controls)
        
        # Show initial state with source+target
        self._show_initial_state()
    
    def _update_bg(self, instance, value):
        """Update background rectangle when widget size/position changes."""
        self.bg.pos = instance.pos
        self.bg.size = instance.size
    
    def go_to_first(self, instance):
        """Go to the first guess."""
        if self.is_running:
            self.pause_visualization(None)
        
        self.current_guess_index = 0
        self._show_initial_state()
    
    def go_back(self, instance):
        """Go back one guess."""
        if self.is_running:
            self.pause_visualization(None)
        
        if self.current_guess_index > 0:
            self.current_guess_index -= 1
            if self.current_guess_index == 0:
                self._show_initial_state()
            else:
                # Show the previous guess
                self._show_specific_guess(self.current_guess_index - 1)
    
    def _show_specific_guess(self, guess_index):
        """Show a specific guess by index."""
        if 0 <= guess_index < len(self.solver_data['guesses']):
            guess = self.solver_data['guesses'][guess_index]
            
            from ...solver.validation.scorer import PlacementScorer
            
            renderer = self.solver_data['renderer']
            scorer = PlacementScorer(overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5)
            
            rendered = renderer.render(guess, self.solver_data['piece_shapes'])
            score = scorer.score(rendered, self.solver_data['target'])
            rendered_color = renderer.render_debug(guess, self.solver_data['piece_shapes'])
            
            if 'puzzle_pieces' in self.solver_data and 'surfaces' in self.solver_data:
                display = self._create_source_target_visualization(
                    rendered_color, 
                    self.solver_data['puzzle_pieces'],
                    self.solver_data['surfaces']
                )
            else:
                display = self._create_visualization(rendered_color, self.solver_data['target'])
            
            self._update_image(display)
            
            is_best = score >= self.solver_data['best_score']
            best_marker = "BEST!" if is_best else ""
            
            self.status_label.text = (
                f'Guess {guess_index + 1}/{len(self.solver_data["guesses"])} | '
                f'Score: {score:.2f}{best_marker}'
            )

    def step_guess(self, instance):
        """Show the next guess."""
        if self.current_guess_index < len(self.solver_data['guesses']):
            guess = self.solver_data['guesses'][self.current_guess_index]
            
            from ...solver.validation.scorer import PlacementScorer
            
            # Use the renderer passed from pipeline
            renderer = self.solver_data['renderer']
            scorer = PlacementScorer(overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5)
            
            # Render grayscale for scoring
            rendered = renderer.render(guess, self.solver_data['piece_shapes'])
            score = scorer.score(rendered, self.solver_data['target'])
            
            # Render in DEBUG mode to show bounding boxes
            rendered_color = renderer.render_debug(guess, self.solver_data['piece_shapes'])
            
            # Create side-by-side visualization with original positions
            if 'puzzle_pieces' in self.solver_data and 'surfaces' in self.solver_data:
                print(f" Using enhanced visualization for guess {self.current_guess_index + 1}")
                display = self._create_source_target_visualization(
                    rendered_color, 
                    self.solver_data['puzzle_pieces'],
                    self.solver_data['surfaces']
                )
            else:
                print(f"⚠️  Using fallback visualization for guess {self.current_guess_index + 1}")
                display = self._create_visualization(rendered_color, self.solver_data['target'])
            
            # Update display
            self._update_image(display)
            
            is_best = score >= self.solver_data['best_score']
            best_marker = " ⭐ NEW BEST!" if is_best else ""
            
            self.status_label.text = (
                f'Guess {self.current_guess_index + 1}/{len(self.solver_data["guesses"])} | '
                f'Score: {score:.2f}{best_marker}'
            )
            
            self.current_guess_index += 1
            
    def show_best(self, instance):
        """Show the best solution found."""
        # Pause if running
        if self.is_running:
            self.pause_visualization(None)
        
        # Get the pre-calculated best guess
        best_guess = self.solver_data.get('best_guess')
        best_guess_index = self.solver_data.get('best_guess_index', 0)
        best_score = self.solver_data.get('best_score', 0)
        
        if best_guess is None:
            self.status_label.text = "No best solution found!"
            return
        
        # Use the renderer passed from pipeline
        renderer = self.solver_data['renderer']
        rendered_color = renderer.render_color(best_guess, self.solver_data['piece_shapes'])
        
        # Create side-by-side visualization
        if 'puzzle_pieces' in self.solver_data and 'surfaces' in self.solver_data:
            print("🎨 Using enhanced visualization for BEST solution")
            display = self._create_source_target_visualization(
                rendered_color,
                self.solver_data['puzzle_pieces'], 
                self.solver_data['surfaces']
            )
        else:
            print("⚠️  Using fallback visualization for BEST solution")
            display = self._create_visualization(rendered_color, self.solver_data['target'])
        
        # Update display
        self._update_image(display)
        
        self.status_label.text = (
            f' BEST SOLUTION | '
            f'Guess #{best_guess_index + 1} | '
            f'Score: {best_score:.2f}'
        )
        
        # Update current index
        self.current_guess_index = best_guess_index

    def _show_initial_state(self):
        """Show initial state with original positions and empty target."""
        if 'puzzle_pieces' in self.solver_data and 'surfaces' in self.solver_data:
            print("🎨 Using enhanced initial state visualization")
            # Create empty guess for target area
            empty_guess = []
            
            # Create empty rendered color (same size as target)
            target = self.solver_data['target']
            empty_rendered = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
            
            # Show source+target visualization with empty target
            display = self._create_source_target_visualization(
                empty_rendered,
                self.solver_data['puzzle_pieces'],
                self.solver_data['surfaces']
            )
        else:
            print("⚠️  Using fallback initial state visualization")
            # Fallback to old target-only view
            target = self.solver_data['target']
            display = (target * 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
            
            # Draw grid lines
            for i in range(0, display.shape[0], 100):
                cv2.line(display, (0, i), (display.shape[1], i), (50, 50, 50), 1)
            for i in range(0, display.shape[1], 100):
                cv2.line(display, (i, 0), (i, display.shape[0]), (50, 50, 50), 1)
        
        self._update_image(display)
        self.status_label.text = f'Initial State | {len(self.solver_data["guesses"])} guesses to test'
    
    def _create_source_target_visualization(self, rendered_color, puzzle_pieces, surfaces):
        """Create side-by-side visualization showing original positions and current guess."""
        print(f"📐 Creating source+target visualization...")
        print(f"   Global: {surfaces['global']['width']}x{surfaces['global']['height']}")
        print(f"   Pieces: {len(puzzle_pieces)}")
        
        # Get global surface dimensions
        global_width = surfaces['global']['width']
        global_height = surfaces['global']['height']
        
        # Create global canvas with light grey background
        canvas = np.full((global_height, global_width, 3), 200, dtype=np.uint8)  # Light grey background
        
        # Get surface offsets
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']  
        target_offset_y = surfaces['target']['offset_y']
        
        # Fill source and target areas with white background
        source_w = surfaces['source']['width']
        source_h = surfaces['source']['height']
        canvas[source_offset_y:source_offset_y + source_h,
               source_offset_x:source_offset_x + source_w] = [255, 255, 255]  # White
        
        target_w = surfaces['target']['width'] 
        target_h = surfaces['target']['height']
        canvas[target_offset_y:target_offset_y + target_h,
               target_offset_x:target_offset_x + target_w] = [255, 255, 255]  # White
        
        # Draw source area boundary (A5) 
        cv2.rectangle(canvas,
                     (source_offset_x, source_offset_y),
                     (source_offset_x + source_w - 1, source_offset_y + source_h - 1),
                     (0, 200, 0), 4)  # Green border (thicker)
        
        # Draw target area boundary (A4)
        cv2.rectangle(canvas,
                     (target_offset_x, target_offset_y),
                     (target_offset_x + target_w - 1, target_offset_y + target_h - 1),
                     (0, 150, 255), 4)  # Orange border (thicker)
        
        # Define piece colors (same as renderer)
        piece_colors = [
            (255, 100, 100),  # Blue-ish
            (100, 255, 100),  # Green-ish  
            (100, 100, 255),  # Red-ish
            (255, 255, 100),  # Cyan-ish
            (255, 100, 255),  # Magenta-ish
            (100, 255, 255),  # Yellow-ish
        ]
        
        # Render original positions (pick_pose) in source area
        print(f"   Rendering {len(puzzle_pieces)} original positions...")
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            x = int(piece.pick_pose.x) + source_offset_x  # Convert to global coords
            y = int(piece.pick_pose.y) + source_offset_y
            theta = piece.pick_pose.theta
            
            print(f"     Piece {piece_id}: ({piece.pick_pose.x}, {piece.pick_pose.y}) -> global ({x}, {y})")
            
            if piece_id in self.solver_data['piece_shapes']:
                shape = self.solver_data['piece_shapes'][piece_id]
                rotated = self._rotate_shape(shape, theta)
                color = piece_colors[piece_id % len(piece_colors)]
                
                # Make original positions semi-transparent
                faded_color = tuple(int(c * 0.7) for c in color)
                self._place_shape_color_global(canvas, rotated, x, y, faded_color)
                
                # Add piece label
                cv2.putText(canvas, f"P{piece_id}", (x + 5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, f"P{piece_id}", (x + 5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Overlay the rendered target area (current guess) 
        target_region = canvas[target_offset_y:target_offset_y + target_h,
                              target_offset_x:target_offset_x + target_w]
        
        # Blend the rendered_color into target region
        if rendered_color.shape[:2] == target_region.shape[:2]:
            # Where rendered_color has content, use it; otherwise keep canvas
            mask = np.any(rendered_color > 0, axis=2)
            target_region[mask] = rendered_color[mask]
            print(f"   Blended target area: {np.sum(mask)} pixels")
        else:
            print(f"   ⚠️  Size mismatch: rendered {rendered_color.shape[:2]} vs target {target_region.shape[:2]}")
        
        # Add area labels with better styling
        cv2.putText(canvas, "A5 SOURCE (Original Positions)", 
                   (source_offset_x + 15, source_offset_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(canvas, "A5 SOURCE (Original Positions)", 
                   (source_offset_x + 15, source_offset_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 0), 2)
        
        cv2.putText(canvas, "A4 TARGET (Current Guess)",
                   (target_offset_x + 15, target_offset_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(canvas, "A4 TARGET (Current Guess)",
                   (target_offset_x + 15, target_offset_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 200), 2)
        
        # Add subtle grid to both areas
        for i in range(0, global_height, 100):
            cv2.line(canvas, (0, i), (global_width, i), (180, 180, 180), 1)
        for i in range(0, global_width, 100):
            cv2.line(canvas, (i, 0), (i, global_height), (180, 180, 180), 1)
        
        print(f"   ✅ Created enhanced visualization: {canvas.shape}")
        return canvas
    
    def _rotate_shape(self, shape: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a shape by angle degrees and crop to tight bounding box."""
        if angle == 0:
            return shape
            
        h, w = shape.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(shape, M, (new_w, new_h))
        
        # Crop to actual content bounds
        piece_points = np.argwhere(rotated > 0)
        if len(piece_points) == 0:
            return rotated
        
        min_y, min_x = piece_points.min(axis=0)
        max_y, max_x = piece_points.max(axis=0)
        
        cropped = rotated[min_y:max_y+1, min_x:max_x+1]
        return cropped
    
    def _place_shape_color_global(self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, color: tuple):
        """Place colored shape on global canvas using TOP-LEFT corner positioning."""
        h, w = shape.shape[:2]
        
        # Calculate bounds - x,y is TOP-LEFT in global coordinates
        y1 = max(0, y)
        y2 = min(canvas.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(canvas.shape[1], x + w)
        
        # Calculate corresponding region in shape
        shape_y1 = max(0, -y)
        shape_y2 = shape_y1 + (y2 - y1)
        shape_x1 = max(0, -x)
        shape_x2 = shape_x1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1 and shape_y2 > shape_y1 and shape_x2 > shape_x1:
            shape_region = shape[shape_y1:shape_y2, shape_x1:shape_x2]
            mask = shape_region > 0
            
            for c in range(3):
                canvas[y1:y2, x1:x2, c][mask] = color[c]

    def start_visualization(self, instance):
        """Start the visualization."""
        if not self.is_running:
            self.is_running = True
            self.start_button.text = 'Playing'
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)
    
    def pause_visualization(self, instance):
        """Pause the visualization."""
        if self.is_running:
            self.is_running = False
            self.start_button.text = 'Play'
            if hasattr(self, 'clock_event'):
                self.clock_event.cancel()
    
    def auto_step(self, dt):
        """Automatically step through guesses."""
        if self.current_guess_index < len(self.solver_data['guesses']):
            self.step_guess(None)
        else:
            self.pause_visualization(None)
            self.status_label.text = f'✅ DONE! Best score: {self.solver_data["best_score"]:.2f}'
    
    def _create_visualization(self, rendered_color, target):
        """Fallback: Create visualization - rendered_color is already in target space."""
        display = rendered_color.copy()
        
        # Draw target outline (which should match the canvas now)
        h, w = display.shape[:2]
        
        # Draw border around entire canvas (which IS the target)
        cv2.rectangle(display, (0, 0), (w-1, h-1), (255, 255, 100), 2)
        
        # Draw grid
        for i in range(0, h, 100):
            cv2.line(display, (0, i), (w, i), (80, 80, 80), 1)
        for i in range(0, w, 100):
            cv2.line(display, (i, 0), (i, h), (80, 80, 80), 1)
        
        return display
    
    def _update_image(self, array: np.ndarray):
        """Update the image widget with a numpy array."""
        # Flip vertically (Kivy uses bottom-left origin)
        display = np.flipud(array)
        
        # Create texture
        texture = Texture.create(size=(display.shape[1], display.shape[0]), colorfmt='rgb')
        texture.blit_buffer(display.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        
        self.image_widget.texture = texture


class SolverVisualizerApp(App):
    def __init__(self, solver_data, **kwargs):
        super().__init__(**kwargs)
        self.solver_data = solver_data
    
    def build(self):
        return SolverVisualizer(self.solver_data)