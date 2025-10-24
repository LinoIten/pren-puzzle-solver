
class Pose:
    def __init__(self, x_mm: float, y_mm: float, theta_deg: float):
        self.x = float(x_mm)
        self.y = float(y_mm)
        self.theta = float(theta_deg)  # [-180, 180]

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.1f} mm, y={self.y:.1f} mm, theta={self.theta:.1f}°)"
