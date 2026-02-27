"""
Movement visualization rendering for the solver visualizer.
Handles COM dots, movement arrows, summary table, and legend.
"""

import cv2
import numpy as np


class MovementRenderer:
    """Renders movement visualization overlays on the solver canvas."""

    def __init__(self, solver_data):
        self.solver_data = solver_data

    def add_com_dots(self, canvas):
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
        self.draw_movement_summary(canvas, movement_summary)

        print(
            f"   ✅ Added COM dots, arrows, and movement data for {len(movement_summary)} pieces"
        )
        return canvas

    def draw_movement_summary(self, canvas, movement_summary):
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

    def draw_movement_legend(self, canvas):
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
