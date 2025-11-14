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

# Update src/ui/simulator/solver_visualizer.py

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
        
        # Show initial state
        self._show_target()
    
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
            
            # Create visualization
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
        
        # Create visualization
        display = self._create_visualization(rendered_color, self.solver_data['target'])
        
        # Update display
        self._update_image(display)
        
        self.status_label.text = (
            f'⭐ BEST SOLUTION ⭐ | '
            f'Guess #{best_guess_index + 1} | '
            f'Score: {best_score:.2f}'
        )
        
        # Update current index
        self.current_guess_index = best_guess_index
    def _show_target(self):
        """Show the target layout initially."""
        target = self.solver_data['target']
        display = (target * 255).astype(np.uint8)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        
        # Draw grid lines
        for i in range(0, display.shape[0], 100):
            cv2.line(display, (0, i), (display.shape[1], i), (50, 50, 50), 1)
        for i in range(0, display.shape[1], 100):
            cv2.line(display, (i, 0), (i, display.shape[0]), (50, 50, 50), 1)
        
        self._update_image(display)
        self.status_label.text = f'Target Layout | {len(self.solver_data["guesses"])} guesses to test'
    
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
        """Create visualization - rendered_color is already in target space."""
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