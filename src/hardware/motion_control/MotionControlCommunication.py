from src.utils.puzzle_piece import PuzzlePiece
import math

def send_to_robot(pieces: list[PuzzlePiece]) -> None:
    """
    Minimal-„Schnittstelle":
    - Wir drucken pro Teil eine Zeile mit x, y, theta.
    - Genau das, was spaeter per seriell/TCP/etc. gesendet werden koennte.
    Format (CSV-aehnlich, sehr lesbar):
      ID; PICK_X_mm; PICK_Y_mm; PICK_THETA_deg; PLACE_X_mm; PLACE_Y_mm; PLACE_THETA_deg
    """
    header = "ID;PICK_X_mm;PICK_Y_mm;PICK_THETA_deg;PLACE_X_mm;PLACE_Y_mm;PLACE_THETA_deg"
    print(header)
    for p in pieces:
        px, py, pt = p.pick_pose.x, p.pick_pose.y, p.pick_pose.theta
        
        # Check if place_pose exists FIRST
        if p.place_pose:
            qx, qy, qt = p.place_pose.x, p.place_pose.y, p.place_pose.theta
        else:
            qx, qy, qt = math.nan, math.nan, math.nan
        
        line = f"{p.id};{px:.1f};{py:.1f};{pt:.1f};{qx:.1f};{qy:.1f};{qt:.1f}"
        print(line)