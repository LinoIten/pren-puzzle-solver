"""
PREN Walking Skeleton – Ein-Datei-Version (nur Standardbibliothek)

Zweck:
- End-to-End-Fluss steht: "Bild" -> Erkennung -> Planung -> Ausgabe an Roboter
- Noch ohne echte Bildverarbeitung, alles simuliert
- Schnittstelle zum Roboter: x, y, theta (Winkel in Grad)

Später ersetzt ihr:
- detect_pieces(...)  -> durch echte OpenCV-Erkennung
- plan_placements(...) -> durch euren Solver
- send_to_robot(...)   -> durch echte Kommunikation (falls gewünscht)

Autor: euer Team
Python: 3.10+
"""

from __future__ import annotations
import math
import random
import time

# -----------------------------
# Konfiguration (einfach halten)
# -----------------------------
WORKSPACE_MM = (0.0, 297.0, 0.0, 210.0)  # (x_min, x_max, y_min, y_max) z.B. A4 in mm
NUM_PIECES   = 4                         # wie viele Puzzleteile wir "finden"
SEED         = 42                        # für reproduzierbare Zufallswerte

# ----------------------------------
# Datentypen (ganz simpel gehalten)
# ----------------------------------
class Pose:
    def __init__(self, x_mm: float, y_mm: float, theta_deg: float):
        self.x = float(x_mm)
        self.y = float(y_mm)
        self.theta = float(theta_deg)  # [-180, 180]

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.1f} mm, y={self.y:.1f} mm, theta={self.theta:.1f}°)"


class Piece:
    def __init__(self, pid: str, pick: Pose):
        self.id = pid
        self.pick_pose = pick
        self.place_pose: Pose | None = None
        self.confidence = 0.0

    def __repr__(self) -> str:
        return (f"Piece(id={self.id}, pick={self.pick_pose}, "
                f"place={self.place_pose}, conf={self.confidence:.2f})")


# --------------------------------------------------
# 1) "Kamerabild" holen (hier nur ein Zeitstempel)
# --------------------------------------------------
def capture_frame() -> dict:
    """
    Platzhalter statt echter Kamera.
    Liefert Metadaten, die man loggen kann.
    """
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_mm": WORKSPACE_MM,
    }


# --------------------------------------------------
# 2) "Erkennung": Teile finden (noch simuliert)
# --------------------------------------------------
def detect_pieces(frame_meta: dict, n: int = NUM_PIECES) -> list[Piece]:
    """
    Platzhalter für OpenCV-Teil:
    - Erzeugt n Teile mit zufälligen (aber reproduzierbaren) Posen.
    - Posen liegen innerhalb des WORKSPACE_MM.
    - Winkel in Grad [-90, 90] (wie minAreaRect oft liefert).
    """
    random.seed(SEED)  # reproduzierbar
    x_min, x_max, y_min, y_max = frame_meta["workspace_mm"]

    pieces: list[Piece] = []
    for i in range(n):
        x = random.uniform(x_min + 20.0, x_max - 20.0)  # 20mm Rand lassen
        y = random.uniform(y_min + 20.0, y_max - 20.0)
        theta = random.uniform(-90.0, 90.0)
        p = Piece(pid=f"p{i+1}", pick=Pose(x, y, theta))
        pieces.append(p)

    # Hinweis: HIER später echte OpenCV-Schritte einbauen:
    # - Graustufen, Threshold, findContours
    # - Schwerpunkt (Momente), Winkel (minAreaRect/PCA)
    return pieces


# --------------------------------------------------
# 3) "Planung": wohin soll das Teil? (noch trivial)
# --------------------------------------------------
def plan_placements(pieces: list[Piece]) -> list[Piece]:
    """
    Platzhalter-Solver:
    - v0: pick == place (wir „tun“ noch nichts)
    - v1 (einfacher Schritt): leg Teile z.B. sauber in eine Reihe am unteren Rand
    """
    # v0: identisch
    for p in pieces:
        p.place_pose = Pose(p.pick_pose.x, p.pick_pose.y, p.pick_pose.theta)
        p.confidence = 0.5  # Dummy-Wert
    return pieces


# --------------------------------------------------
# 4) "Roboter-Schnittstelle": Koordinaten ausgeben
# --------------------------------------------------
def send_to_robot(pieces: list[Piece]) -> None:
    """
    Minimal-„Schnittstelle“:
    - Wir drucken pro Teil eine Zeile mit x, y, theta.
    - Genau das, was später per seriell/TCP/etc. gesendet werden könnte.
    Format (CSV-ähnlich, sehr lesbar):
      ID; PICK_X_mm; PICK_Y_mm; PICK_THETA_deg; PLACE_X_mm; PLACE_Y_mm; PLACE_THETA_deg
    """
    header = "ID;PICK_X_mm;PICK_Y_mm;PICK_THETA_deg;PLACE_X_mm;PLACE_Y_mm;PLACE_THETA_deg"
    print(header)
    for p in pieces:
        px, py, pt = p.pick_pose.x, p.pick_pose.y, p.pick_pose.theta
        qx, qy, qt = p.place_pose.x, p.place_pose.y, p.place_pose.theta if p.place_pose else (math.nan, math.nan, math.nan)
        line = f"{p.id};{px:.1f};{py:.1f};{pt:.1f};{qx:.1f};{qy:.1f};{qt:.1f}"
        print(line)


# --------------------------------------------------
# (Optional) Mini-Check: sind die Posen plausibel?
# --------------------------------------------------
def sanity_check(pieces: list[Piece]) -> None:
    """
    Prüft, ob Posen im Workspace liegen; druckt Warnungen, bricht aber nicht ab.
    """
    x_min, x_max, y_min, y_max = WORKSPACE_MM
    for p in pieces:
        for label, pose in (("pick", p.pick_pose), ("place", p.place_pose or p.pick_pose)):
            if not (x_min <= pose.x <= x_max) or not (y_min <= pose.y <= y_max):
                print(f"[WARN] {p.id} {label}-Pose außerhalb des Arbeitsbereichs: {pose}")


# --------------------------------------------------
# 5) Orchestrierung
# --------------------------------------------------
def main() -> None:
    print("=== PREN Walking Skeleton (Ein-Datei) ===")
    frame = capture_frame()
    print(f"[INFO] Frame @ {frame['timestamp']} – Workspace (mm): {frame['workspace_mm']}")

    pieces = detect_pieces(frame, n=NUM_PIECES)
    print(f"[INFO] Erkannt: {len(pieces)} Teil(e)")

    planned = plan_placements(pieces)
    sanity_check(planned)

    print("\n--- Ausgabe an Roboter (x, y, theta in mm/Grad) ---")
    send_to_robot(planned)
    print("\n[OK] Ende-zu-Ende-Pfad steht. Jetzt könnt ihr die Platzhalter schrittweise ersetzen.")


if __name__ == "__main__":
    main()
