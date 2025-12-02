# src/ui/solver_visualizer.py

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import numpy as np
import cv2

class SolverVisualizer(BoxLayout):
    def __init__(self, solver_data, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.solver_data = solver_data
        self.current_guess_index = 0
        self.is_running = False
        self.speed = 0.05  # seconds per guess
        
        # Top: Image display
        self.image_widget = Image(size_hint_y=0.8)
        self.add_widget(self.image_widget)
        
        # Bottom: Controls
        controls = BoxLayout(size_hint_y=0.2, orientation='vertical', padding=10, spacing=10)
        
        # Status label
        self.status_label = Label(text='Ready to visualize', size_hint_y=0.3)
        controls.add_widget(self.status_label)
        
        # Buttons - First row
        button_row1 = BoxLayout(orientation='horizontal', size_hint_y=0.35, spacing=10)
        
        self.start_button = Button(text='Start')
        self.start_button.bind(on_press=self.start_visualization)  # type: ignore
        button_row1.add_widget(self.start_button)
        
        self.pause_button = Button(text='Pause')
        self.pause_button.bind(on_press=self.pause_visualization) # type: ignore
        button_row1.add_widget(self.pause_button)
        
        self.step_button = Button(text='Next')
        self.step_button.bind(on_press=self.step_guess) # type: ignore
        button_row1.add_widget(self.step_button)
        
        self.best_button = Button(text='Show Best', background_color=(0.2, 0.8, 0.2, 1))
        self.best_button.bind(on_press=self.show_best) # type: ignore
        button_row1.add_widget(self.best_button)
        
        controls.add_widget(button_row1)
        
        # Buttons - Second row
        button_row2 = BoxLayout(orientation='horizontal', size_hint_y=0.35, spacing=10)
        
        self.speed_up_button = Button(text='Speed++')
        self.speed_up_button.bind(on_press=self.speed_up) # type: ignore
        button_row2.add_widget(self.speed_up_button)
        
        self.speed_down_button = Button(text='Speed--')
        self.speed_down_button.bind(on_press=self.speed_down) # type: ignore
        button_row2.add_widget(self.speed_down_button)
        
        controls.add_widget(button_row2)
        
        # Speed label
        self.speed_label = Label(text=f'Speed: {1/self.speed:.0f} guesses/sec', size_hint_y=0.35)
        controls.add_widget(self.speed_label)
        
        self.add_widget(controls)
        
        # Show initial state with source+target
        self._show_initial_state()
    
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
                display = self._create_source_target_visualization(
                    rendered_color, 
                    self.solver_data['puzzle_pieces'],
                    self.solver_data['surfaces']
                )
            else:
                # Fallback to old visualization
                display = self._create_visualization(rendered_color, self.solver_data['target'])
            
            # Update display
            self._update_image(display)
            
            is_best = score >= self.solver_data['best_score']
            best_marker = " NEW BEST!" if is_best else ""
            
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
            display = self._create_source_target_visualization(
                rendered_color,
                self.solver_data['puzzle_pieces'], 
                self.solver_data['surfaces']
            )
        else:
            # Fallback to old visualization
            display = self._create_visualization(rendered_color, self.solver_data['target'])
        
        # Update display
        self._update_image(display)
        
        self.status_label.text = (
            f' BEST SOLUTION  | '
            f'Guess #{best_guess_index + 1} | '
            f'Score: {best_score:.2f}'
        )
        
        # Update current index
        self.current_guess_index = best_guess_index

    def _show_initial_state(self):
        """Show initial state with original positions and empty target."""
        if 'puzzle_pieces' in self.solver_data and 'surfaces' in self.solver_data:
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
        # Get global surface dimensions
        global_width = surfaces['global']['width']
        global_height = surfaces['global']['height']
        
        # Create global canvas
        canvas = np.zeros((global_height, global_width, 3), dtype=np.uint8)
        
        # Get surface offsets
        source_offset_x = surfaces['source']['offset_x']
        source_offset_y = surfaces['source']['offset_y']
        target_offset_x = surfaces['target']['offset_x']  
        target_offset_y = surfaces['target']['offset_y']
        
        # Draw source area boundary (A5) 
        source_w = surfaces['source']['width']
        source_h = surfaces['source']['height']
        cv2.rectangle(canvas,
                     (source_offset_x, source_offset_y),
                     (source_offset_x + source_w - 1, source_offset_y + source_h - 1),
                     (0, 255, 0), 2)  # Green border
        
        # Draw target area boundary (A4)
        target_w = surfaces['target']['width'] 
        target_h = surfaces['target']['height']
        cv2.rectangle(canvas,
                     (target_offset_x, target_offset_y),
                     (target_offset_x + target_w - 1, target_offset_y + target_h - 1),
                     (0, 255, 255), 2)  # Yellow border
        
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
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            x = int(piece.pick_pose.x) + source_offset_x  # Convert to global coords
            y = int(piece.pick_pose.y) + source_offset_y
            theta = piece.pick_pose.theta
            
            if piece_id in self.solver_data['piece_shapes']:
                shape = self.solver_data['piece_shapes'][piece_id]
                rotated = self._rotate_shape(shape, theta)
                color = piece_colors[piece_id % len(piece_colors)]
                
                # Make original positions semi-transparent
                faded_color = tuple(int(c * 0.6) for c in color)
                self._place_shape_color_global(canvas, rotated, x, y, faded_color)
                
                # Add "PICK" label
                cv2.putText(canvas, f"P{piece_id}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Overlay the rendered target area (current guess) 
        target_region = canvas[target_offset_y:target_offset_y + target_h,
                              target_offset_x:target_offset_x + target_w]
        
        # Blend the rendered_color into target region
        if rendered_color.shape[:2] == target_region.shape[:2]:
            # Where rendered_color has content, use it; otherwise keep canvas
            mask = np.any(rendered_color > 0, axis=2)
            target_region[mask] = rendered_color[mask]
        
        # Add area labels
        cv2.putText(canvas, "A5 SOURCE (Original)", 
                   (source_offset_x + 10, source_offset_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(canvas, "A4 TARGET (Current Guess)",
                   (target_offset_x + 10, target_offset_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add grid to both areas
        for i in range(0, global_height, 50):
            cv2.line(canvas, (0, i), (global_width, i), (40, 40, 40), 1)
        for i in range(0, global_width, 50):
            cv2.line(canvas, (i, 0), (i, global_height), (40, 40, 40), 1)
        
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
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)
    
    def pause_visualization(self, instance):
        """Pause the visualization."""
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'clock_event'):
                self.clock_event.cancel()
    
    def speed_up(self, instance):
        """Speed up the visualization."""
        self.speed = max(0.001, self.speed / 2)
        self.speed_label.text = f'Speed: {1/self.speed:.0f} guesses/sec'
        if self.is_running:
            self.clock_event.cancel()
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)
    
    def speed_down(self, instance):
        """Slow down the visualization."""
        self.speed = min(2.0, self.speed * 2)
        self.speed_label.text = f'Speed: {1/self.speed:.0f} guesses/sec'
        if self.is_running:
            self.clock_event.cancel()
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)
    
    def auto_step(self, dt):
        """Automatically step through guesses."""
        if self.current_guess_index < len(self.solver_data['guesses']):
            self.step_guess(None)
        else:
            self.pause_visualization(None)
            self.status_label.text = f'DONE! Best score: {self.solver_data["best_score"]:.2f}'
    
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
        # Flip vertically (Kivy uses bottomfrom src.ui.simulator.solver_vi-left origin)
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