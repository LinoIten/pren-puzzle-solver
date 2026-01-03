import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from ...solver.validation.scorer import PlacementScorer


class SolverVisualizer(BoxLayout):
    def __init__(self, solver_data, embedded=False, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.solver_data = solver_data
        self.embedded = embedded  # New parameter
        self.current_guess_index = 0
        self.is_running = False
        self.speed = 0.05
        self.show_movements = False

        # Only add background if not embedded
        if not self.embedded:
            with self.canvas.before:
                Color(0.9, 0.9, 0.9, 1)
                self.bg = Rectangle(size=self.size, pos=self.pos)
                self.bind(size=self._update_bg, pos=self._update_bg)

        # Top: Image display
        self.image_widget = Image(size_hint_y=0.75)
        self.add_widget(self.image_widget)

        # Bottom: Controls with better styling
        controls = BoxLayout(
            size_hint_y=0.25, orientation="vertical", padding=20, spacing=15
        )

        # Status label with better styling
        self.status_label = Label(
            text="Ready to visualize",
            size_hint_y=0.2,
            color=(0.2, 0.2, 0.2, 1),  # Dark grey text
            font_size="16sp",
            bold=True,
        )
        controls.add_widget(self.status_label)

        # Main controls - First row (Navigation & Playback)
        button_row1 = BoxLayout(orientation="horizontal", size_hint_y=0.4, spacing=10)

        # Navigation buttons
        self.first_button = Button(
            text="️First",
            background_color=(0.3, 0.5, 0.8, 1),  # Blue
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.first_button.bind(on_press=self.go_to_first)
        button_row1.add_widget(self.first_button)

        self.back_button = Button(
            text="Back",
            background_color=(0.7, 0.5, 0.2, 1),  # Orange
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.back_button.bind(on_press=self.go_back)
        button_row1.add_widget(self.back_button)

        self.step_button = Button(
            text="Next",
            background_color=(0.4, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.step_button.bind(on_press=self.step_guess)
        button_row1.add_widget(self.step_button)

        # Playback controls
        self.start_button = Button(
            text="Play",
            background_color=(0.3, 0.7, 0.3, 1),  # Green
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.start_button.bind(on_press=self.start_visualization)
        button_row1.add_widget(self.start_button)

        self.pause_button = Button(
            text="Pause",
            background_color=(0.8, 0.5, 0.2, 1),  # Orange-red
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.pause_button.bind(on_press=self.pause_visualization)
        button_row1.add_widget(self.pause_button)

        self.best_button = Button(
            text="Best",
            background_color=(0.8, 0.2, 0.8, 1),  # Purple
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.best_button.bind(on_press=self.show_best)
        button_row1.add_widget(self.best_button)

        self.show_movement_button = Button(
            text="Movement",
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.show_movement_button.bind(on_press=self.toggle_movement_view)
        button_row1.add_widget(self.show_movement_button)

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
        if 0 <= guess_index < len(self.solver_data["guesses"]):
            guess = self.solver_data["guesses"][guess_index]

            renderer = self.solver_data["renderer"]
            scorer = PlacementScorer(
                overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5
            )

            rendered = renderer.render(guess, self.solver_data["piece_shapes"])
            score = scorer.score(rendered, self.solver_data["target"])
            rendered_color = renderer.render_debug(
                guess, self.solver_data["piece_shapes"]
            )

            if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
                display = self._create_source_target_visualization(
                    rendered_color,
                    self.solver_data["puzzle_pieces"],
                    self.solver_data["surfaces"],
                )
            else:
                display = self._create_visualization(
                    rendered_color, self.solver_data["target"]
                )

            self._update_image(display)

            is_best = score >= self.solver_data["best_score"]
            best_marker = "BEST!" if is_best else ""

            self.status_label.text = (
                f"Guess {guess_index + 1}/{len(self.solver_data['guesses'])} | "
                f"Score: {score:.2f}{best_marker}"
            )

    def step_guess(self, instance):
        """Show the next guess."""
        if self.current_guess_index < len(self.solver_data["guesses"]):
            guess = self.solver_data["guesses"][self.current_guess_index]

            renderer = self.solver_data["renderer"]
            scorer = PlacementScorer(
                overlap_penalty=2.0, coverage_reward=1.0, gap_penalty=0.5
            )

            rendered = renderer.render(guess, self.solver_data["piece_shapes"])
            score = scorer.score(rendered, self.solver_data["target"])

            rendered_color = renderer.render_debug(
                guess, self.solver_data["piece_shapes"]
            )

            if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
                display = self._create_source_target_visualization(
                    rendered_color,
                    self.solver_data["puzzle_pieces"],
                    self.solver_data["surfaces"],
                )
            else:
                display = self._create_visualization(
                    rendered_color, self.solver_data["target"]
                )

            self._update_image(display)

            is_best = score >= self.solver_data["best_score"]
            best_marker = " NEW BEST!" if is_best else ""

            self.status_label.text = (
                f"Guess {self.current_guess_index + 1}/{len(self.solver_data['guesses'])} | "
                f"Score: {score:.2f}{best_marker}"
            )

            self.current_guess_index += 1

    def show_best(self, instance):
        """Show the best solution found."""
        # Pause if running
        if self.is_running:
            self.pause_visualization(None)

        # Get the pre-calculated best guess
        best_guess = self.solver_data.get("best_guess")
        best_guess_index = self.solver_data.get("best_guess_index", 0)
        best_score = self.solver_data.get("best_score", 0)

        if best_guess is None:
            self.status_label.text = "No best solution found!"
            return

        # Use the renderer passed from pipeline
        renderer = self.solver_data["renderer"]
        rendered_color = renderer.render_color(
            best_guess, self.solver_data["piece_shapes"]
        )

        # Create side-by-side visualization
        if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
            display = self._create_source_target_visualization(
                rendered_color,
                self.solver_data["puzzle_pieces"],
                self.solver_data["surfaces"],
            )
        else:
            print("⚠️  Using fallback visualization for BEST solution")
            display = self._create_visualization(
                rendered_color, self.solver_data["target"]
            )

        # Update display
        self._update_image(display)

        self.status_label.text = (
            f" BEST SOLUTION | Guess #{best_guess_index + 1} | Score: {best_score:.2f}"
        )

        # Update current index
        self.current_guess_index = best_guess_index

    def _show_initial_state(self):
        """Show initial state with original positions and empty target."""
        if "puzzle_pieces" in self.solver_data and "surfaces" in self.solver_data:
            # Create empty guess for target area
            empty_guess = []

            # Create empty rendered color (same size as target)
            target = self.solver_data["target"]
            empty_rendered = np.zeros(
                (target.shape[0], target.shape[1], 3), dtype=np.uint8
            )

            # Show source+target visualization with empty target
            display = self._create_source_target_visualization(
                empty_rendered,
                self.solver_data["puzzle_pieces"],
                self.solver_data["surfaces"],
            )
        else:
            print("⚠️  Using fallback initial state visualization")
            # Fallback to old target-only view
            target = self.solver_data["target"]
            display = (target * 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

            # Draw grid lines
            for i in range(0, display.shape[0], 100):
                cv2.line(display, (0, i), (display.shape[1], i), (50, 50, 50), 1)
            for i in range(0, display.shape[1], 100):
                cv2.line(display, (i, 0), (i, display.shape[0]), (50, 50, 50), 1)

        self._update_image(display)
        self.status_label.text = (
            f"Initial State | {len(self.solver_data['guesses'])} guesses to test"
        )

    def _create_source_target_visualization(
        self, rendered_color, puzzle_pieces, surfaces, show_movements=None
    ):
        """Create side-by-side visualization with optional COM dots."""
        # Use instance variable if not specified
        if show_movements is None:
            show_movements = getattr(self, "show_movements", False)

        print(f"📐 Creating visualization (movements: {show_movements})...")

        global_width = surfaces["global"]["width"]
        global_height = surfaces["global"]["height"]
        canvas = np.full((global_height, global_width, 3), 200, dtype=np.uint8)

        source_offset_x = surfaces["source"]["offset_x"]
        source_offset_y = surfaces["source"]["offset_y"]
        target_offset_x = surfaces["target"]["offset_x"]
        target_offset_y = surfaces["target"]["offset_y"]

        # Fill areas white, draw borders
        source_w, source_h = surfaces["source"]["width"], surfaces["source"]["height"]
        target_w, target_h = surfaces["target"]["width"], surfaces["target"]["height"]

        canvas[
            source_offset_y : source_offset_y + source_h,
            source_offset_x : source_offset_x + source_w,
        ] = [255, 255, 255]
        canvas[
            target_offset_y : target_offset_y + target_h,
            target_offset_x : target_offset_x + target_w,
        ] = [255, 255, 255]

        cv2.rectangle(
            canvas,
            (source_offset_x, source_offset_y),
            (source_offset_x + source_w - 1, source_offset_y + source_h - 1),
            (0, 200, 0),
            4,
        )
        cv2.rectangle(
            canvas,
            (target_offset_x, target_offset_y),
            (target_offset_x + target_w - 1, target_offset_y + target_h - 1),
            (0, 150, 255),
            4,
        )

        piece_colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
        ]

        # Render original positions in source area
        for piece in puzzle_pieces:
            piece_id = int(piece.id)
            x = int(piece.pick_pose.x) + source_offset_x
            y = int(piece.pick_pose.y) + source_offset_y

            if piece_id in self.solver_data["piece_shapes"]:
                shape = self.solver_data["piece_shapes"][piece_id]
                rotated = self._rotate_shape(shape, piece.pick_pose.theta)
                color = piece_colors[piece_id % len(piece_colors)]
                faded_color = tuple(int(c * 0.7) for c in color)

                self._place_shape_color_global(canvas, rotated, x, y, faded_color)
                cv2.putText(
                    canvas,
                    f"P{piece_id}",
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    canvas,
                    f"P{piece_id}",
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )

        # Overlay target area
        target_region = canvas[
            target_offset_y : target_offset_y + target_h,
            target_offset_x : target_offset_x + target_w,
        ]
        if rendered_color.shape[:2] == target_region.shape[:2]:
            mask = np.any(rendered_color > 0, axis=2)
            target_region[mask] = rendered_color[mask]

        # Add COM dots if requested
        if show_movements and "movement_data" in self.solver_data:
            canvas = self._add_com_dots(canvas)

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

        cropped = rotated[min_y : max_y + 1, min_x : max_x + 1]
        return cropped

    def _place_shape_color_global(
        self, canvas: np.ndarray, shape: np.ndarray, x: int, y: int, color: tuple
    ):
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
            self.start_button.text = "Playing"
            self.clock_event = Clock.schedule_interval(self.auto_step, self.speed)

    def pause_visualization(self, instance):
        """Pause the visualization."""
        if self.is_running:
            self.is_running = False
            self.start_button.text = "Play"
            if hasattr(self, "clock_event"):
                self.clock_event.cancel()

    def auto_step(self, dt):
        """Automatically step through guesses."""
        if self.current_guess_index < len(self.solver_data["guesses"]):
            self.step_guess(None)
        else:
            self.pause_visualization(None)
            self.status_label.text = (
                f"✅ DONE! Best score: {self.solver_data['best_score']:.2f}"
            )

    def _create_visualization(self, rendered_color, target):
        """Fallback: Create visualization - rendered_color is already in target space."""
        display = rendered_color.copy()

        # Draw target outline (which should match the canvas now)
        h, w = display.shape[:2]

        # Draw border around entire canvas (which IS the target)
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (255, 255, 100), 2)

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
        texture = Texture.create(
            size=(display.shape[1], display.shape[0]), colorfmt="rgb"
        )
        texture.blit_buffer(display.tobytes(), colorfmt="rgb", bufferfmt="ubyte")

        self.image_widget.texture = texture

    def _add_com_dots(self, canvas):
        """Add COM dots, movement arrows, and movement values to existing canvas."""
        movement_data = self.solver_data["movement_data"]
        source_coms = movement_data.get("source_coms", {})
        target_coms = movement_data.get("target_coms", {})
        movements = movement_data.get("movements", {})

        print(f"🎯 Adding COM visualization for {len(source_coms)} pieces...")

        movement_summary = []  # Collect movement data for display

        for piece_id in source_coms:
            source_com = source_coms[piece_id]

            # Source COM (yellow with S)
            cv2.circle(
                canvas, (int(source_com[0]), int(source_com[1])), 8, (0, 0, 0), -1
            )
            cv2.circle(
                canvas, (int(source_com[0]), int(source_com[1])), 6, (255, 255, 0), -1
            )
            cv2.putText(
                canvas,
                "S",
                (int(source_com[0]) - 4, int(source_com[1]) + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
            )

            # Target COM and movement arrow
            if piece_id in target_coms and piece_id in movements:
                target_com = target_coms[piece_id]
                movement = movements[piece_id]

                # Target COM (cyan with T)
                cv2.circle(
                    canvas, (int(target_com[0]), int(target_com[1])), 8, (0, 0, 0), -1
                )
                cv2.circle(
                    canvas,
                    (int(target_com[0]), int(target_com[1])),
                    6,
                    (0, 255, 255),
                    -1,
                )
                cv2.putText(
                    canvas,
                    "T",
                    (int(target_com[0]) - 4, int(target_com[1]) + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                )

                # Calculate movement components
                dx = target_com[0] - source_com[0]  # X movement (positive = right)
                dy = target_com[1] - source_com[1]  # Y movement (positive = down)
                distance = movement["distance"]
                rotation = movement["rotation"]

                # Store movement summary for display
                movement_summary.append(
                    {
                        "piece_id": piece_id,
                        "dx": dx,
                        "dy": dy,
                        "distance": distance,
                        "rotation": rotation,
                    }
                )

                # Draw movement arrow (only if significant movement > 10px)
                if distance > 10:
                    start_point = (int(source_com[0]), int(source_com[1]))
                    end_point = (int(target_com[0]), int(target_com[1]))

                    # Draw thick arrow
                    cv2.arrowedLine(
                        canvas, start_point, end_point, (50, 50, 200), 4, tipLength=0.1
                    )

                    # Add distance label at midpoint of arrow
                    mid_x = int((source_com[0] + target_com[0]) / 2)
                    mid_y = int((source_com[1] + target_com[1]) / 2)

                    dist_text = f"{distance:.0f}px"
                    (text_w, text_h), _ = cv2.getTextSize(
                        dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    # White background for distance text
                    cv2.rectangle(
                        canvas,
                        (mid_x - text_w // 2 - 3, mid_y - text_h - 3),
                        (mid_x + text_w // 2 + 3, mid_y + 3),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (mid_x - text_w // 2 - 3, mid_y - text_h - 3),
                        (mid_x + text_w // 2 + 3, mid_y + 3),
                        (0, 0, 0),
                        1,
                    )

                    cv2.putText(
                        canvas,
                        dist_text,
                        (mid_x - text_w // 2, mid_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (50, 50, 200),
                        1,
                    )

                # Draw rotation indicator at target position
                if abs(rotation) > 5:  # Only show significant rotations
                    center = (int(target_com[0]), int(target_com[1]))
                    radius = 20

                    # Draw rotation arc
                    if rotation > 0:
                        # Clockwise rotation (red)
                        cv2.ellipse(
                            canvas,
                            center,
                            (radius, radius),
                            0,
                            0,
                            min(90, abs(rotation)),
                            (0, 0, 200),
                            2,
                        )
                        rot_symbol = f"↻{rotation:.0f}°"
                    else:
                        # Counter-clockwise rotation (blue)
                        cv2.ellipse(
                            canvas,
                            center,
                            (radius, radius),
                            0,
                            max(-90, rotation),
                            0,
                            (200, 0, 0),
                            2,
                        )
                        rot_symbol = f"↺{abs(rotation):.0f}°"

                    # Add rotation text near target
                    cv2.putText(
                        canvas,
                        rot_symbol,
                        (center[0] + 25, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                    )

        # Add comprehensive movement summary at bottom
        self._draw_movement_summary(canvas, movement_summary)

        # Add legend at top
        # self._draw_movement_legend(canvas)

        print(
            f"   ✅ Added COM dots, arrows, and movement data for {len(movement_summary)} pieces"
        )
        return canvas

    def _draw_movement_summary(self, canvas, movement_summary):
        """Draw detailed movement summary at bottom of canvas - clean format for robot."""
        if not movement_summary:
            return

        # Position summary at bottom center
        canvas_height, canvas_width = canvas.shape[:2]
        summary_x = 20
        summary_y = canvas_height - 120
        summary_width = canvas_width - 40
        summary_height = 100

        # Background box
        cv2.rectangle(
            canvas,
            (summary_x, summary_y),
            (summary_x + summary_width, summary_y + summary_height),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (summary_x, summary_y),
            (summary_x + summary_width, summary_y + summary_height),
            (0, 0, 0),
            2,
        )

        # Title
        """cv2.putText(canvas, "ROBOT MOVEMENT INSTRUCTIONS",
                (summary_x + 10, summary_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Column headers
        cv2.putText(canvas, "Piece", (summary_x + 10, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, "X Move (mm)", (summary_x + 60, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, "Y Move (mm)", (summary_x + 150, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, "Distance (mm)", (summary_x + 240, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, "Rotation (deg)", (summary_x + 340, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw header line
        cv2.line(canvas, (summary_x + 5, header_y + 5), (summary_x + summary_width - 10, header_y + 5), (0, 0, 0), 1)
        """
        header_y = summary_y + 40
        # Movement data for each piece
        for i, movement in enumerate(movement_summary):
            data_y = header_y + 20 + (i * 18)

            if data_y > summary_y + summary_height - 10:  # Don't overflow box
                break

            piece_id = movement["piece_id"]

            # Get movement data from the movement dictionary
            movement_data = self.solver_data["movement_data"]["movements"].get(
                piece_id, {}
            )
            x_mm = movement_data.get("x_mm", 0)
            y_mm = movement_data.get("y_mm", 0)
            distance_mm = movement_data.get("distance_mm", 0)
            rotation = movement_data.get("rotation", 0)

            # Convert rotation to 0-360 range
            rotation_360 = rotation % 360

            # Piece ID
            cv2.putText(
                canvas,
                f"P{piece_id}",
                (summary_x + 15, data_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            # X Movement in mm
            x_text = f"{x_mm:+.1f}" if abs(x_mm) > 0.1 else "0.0"
            cv2.putText(
                canvas,
                x_text,
                (summary_x + 75, data_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            # Y Movement in mm
            y_text = f"{y_mm:+.1f}" if abs(y_mm) > 0.1 else "0.0"
            cv2.putText(
                canvas,
                y_text,
                (summary_x + 165, data_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            # Total Distance in mm
            cv2.putText(
                canvas,
                f"{distance_mm:.1f}",
                (summary_x + 260, data_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            # Rotation in 0-360 degrees
            cv2.putText(
                canvas,
                f"{rotation_360:.0f}",
                (summary_x + 365, data_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    def _draw_movement_legend(self, canvas):
        """Draw legend explaining movement visualization symbols."""
        legend_x = 20
        legend_y = 80
        legend_width = 450
        legend_height = 80

        # Background
        cv2.rectangle(
            canvas,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (0, 0, 0),
            1,
        )

        # Title
        cv2.putText(
            canvas,
            "Movement Legend:",
            (legend_x + 10, legend_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # COM dots
        y_line1 = legend_y + 35
        cv2.circle(canvas, (legend_x + 20, y_line1), 6, (255, 255, 0), -1)
        cv2.putText(
            canvas,
            "Source COM",
            (legend_x + 30, y_line1 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

        cv2.circle(canvas, (legend_x + 120, y_line1), 6, (0, 255, 255), -1)
        cv2.putText(
            canvas,
            "Target COM",
            (legend_x + 130, y_line1 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

        # Movement arrow
        arrow_start = (legend_x + 220, y_line1)
        arrow_end = (legend_x + 250, y_line1)
        cv2.arrowedLine(canvas, arrow_start, arrow_end, (50, 50, 200), 3, tipLength=0.2)
        cv2.putText(
            canvas,
            "Movement",
            (legend_x + 260, y_line1 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

        # Rotation indicators
        y_line2 = legend_y + 55
        cv2.putText(
            canvas,
            "Rotation:",
            (legend_x + 10, y_line2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            canvas,
            "↻ Clockwise",
            (legend_x + 80, y_line2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 200),
            1,
        )
        cv2.putText(
            canvas,
            "↺ Counter-CW",
            (legend_x + 180, y_line2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 0, 0),
            1,
        )

        # Movement directions
        cv2.putText(
            canvas,
            "→←↑↓ Direction",
            (legend_x + 290, y_line2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    def add_movement_button_to_init(self):
        """Add this button to your button_row1 in __init__"""
        self.show_movement_button = Button(
            text="🎯 Movement",
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
            font_size="14sp",
            bold=True,
        )
        self.show_movement_button.bind(on_press=self.toggle_movement_view)
        # button_row1.add_widget(self.show_movement_button)
        self.show_movements = False  # Track state

    def toggle_movement_view(self, instance):
        """Toggle movement visualization and jump to best solution."""
        if not self.solver_data.get("movement_data"):
            self.status_label.text = "No movement data available!"
            return

        self.show_movements = not self.show_movements
        self.show_movement_button.text = (
            "🎯 Hide Movement" if self.show_movements else "🎯 Movement"
        )
        self.show_movement_button.background_color = (
            (0.8, 0.4, 0.2, 1) if self.show_movements else (0.2, 0.6, 0.8, 1)
        )
        self.show_best(None)  # Jump to best solution


# app
class SolverVisualizerApp(App):
    def __init__(self, solver_data, **kwargs):
        super().__init__(**kwargs)
        self.solver_data = solver_data

    def build(self):
        visualizer = SolverVisualizer(self.solver_data)
        visualizer.add_movement_button_to_init()
        return visualizer
