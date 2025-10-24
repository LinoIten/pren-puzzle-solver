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

# -----------------------------
# Konfiguration (einfach halten)
# -----------------------------
WORKSPACE_MM = (0.0, 297.0, 0.0, 210.0)  # (x_min, x_max, y_min, y_max) z.B. A4 in mm
NUM_PIECES   = 4                         # wie viele Puzzleteile wir "finden"
SEED         = 42                        # für reproduzierbare Zufallswerte

# ----------------------------------
# Datentypen (ganz simpel gehalten)
# ----------------------------------




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
