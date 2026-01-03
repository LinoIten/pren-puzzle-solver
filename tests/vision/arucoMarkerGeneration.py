# generate_aruco_a4_corners_pdf.py
#
# Erzeugt eine druckfertige A4-PDF, in der 4 ArUco-Marker (IDs 0-3, DICT_4X4_50)
# bereits an den A4-ECKEN platziert sind, allerdings mit einem kleinen Seitenabstand
# !!-> Erzeugt wegen diesem Abstand eine leichte Verzerrung->Teil wird zu gross (für Testing ok)
#
# WICHTIG beim Drucken:
# - Skalierung: 100%


import os
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


OUTPUT_PDF = "aruco_A4_corners_ids_0_1_2_3.pdf"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]

#45, 50...
MARKER_SIZE_MM = 45

# Abstand des Markers von der Papierkante (in mm).
# Damit der Drucker nicht abschneidet
EDGE_MARGIN_MM = 5

# Auflösung der Marker-Bitmap, höher ist sauberer beim drucken
MARKER_BITMAP_PX = 900

# Beschriftung neben den Markern
DRAW_LABELS = True
LABEL_FONT_SIZE = 10


def main():
    page_w_pt, page_h_pt = A4  # in points
    page_w_mm = page_w_pt / mm
    page_h_mm = page_h_pt / mm

    # Sicherstellen: Marker passt mit Rand
    needed = 2 * EDGE_MARGIN_MM + MARKER_SIZE_MM
    if needed > page_w_mm or needed > page_h_mm:
        raise ValueError(
            f"Marker+Rand passt nicht auf A4. "
            f"needed={needed:.1f}mm, A4={page_w_mm:.1f}x{page_h_mm:.1f}mm"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # Eckpositionen (x_mm, y_mm) = UNTERE LINKE Ecke des Marker-Bildes in mm (PDF-Koordinaten)
    # PDF-Koordinaten: Ursprung unten links.
    x_left = EDGE_MARGIN_MM
    x_right = page_w_mm - EDGE_MARGIN_MM - MARKER_SIZE_MM
    y_bottom = EDGE_MARGIN_MM
    y_top = page_h_mm - EDGE_MARGIN_MM - MARKER_SIZE_MM

    # Hochformat A4:
    # ID 0: oben links, ID 1: oben rechts, ID 2: unten rechts, ID 3: unten links
    placements = {
        0: (x_left,  y_top),
        1: (x_right, y_top),
        2: (x_right, y_bottom),
        3: (x_left,  y_bottom),
    }

    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)

    # Optional Hinweistext in der Mitte
    c.setFont("Helvetica", 12)
    c.drawString(EDGE_MARGIN_MM * mm, (page_h_mm / 2) * mm,
                 "ArUco-Eckenmarker A4 (IDs 0..3) — Druck: 100% / keine Skalierung")

    tmp_files = []
    try:
        for marker_id in MARKER_IDS:
            marker_img = cv2.aruco.generateImageMarker(
                aruco_dict, marker_id, MARKER_BITMAP_PX
            )

            tmp_name = f"_tmp_aruco_{marker_id}.png"
            cv2.imwrite(tmp_name, marker_img)
            tmp_files.append(tmp_name)

            x_mm, y_mm = placements[marker_id]

            c.drawImage(
                tmp_name,
                x_mm * mm,
                y_mm * mm,
                MARKER_SIZE_MM * mm,
                MARKER_SIZE_MM * mm,
            )

            if DRAW_LABELS:
                c.setFont("Helvetica", LABEL_FONT_SIZE)
                # Label leicht "innen" neben den Marker setzen, damit nichts abgeschnitten wird
                label = f"ID {marker_id}"
                # Position je Ecke sinnvoll verschieben
                if marker_id in (0, 3):  # links
                    lx = (x_mm + MARKER_SIZE_MM + 2) * mm
                else:  # rechts
                    lx = (x_mm - 18) * mm
                if marker_id in (0, 1):  # oben
                    ly = (y_mm + MARKER_SIZE_MM + 2) * mm
                else:  # unten
                    ly = (y_mm - 6) * mm
                c.drawString(lx, ly, label)

        c.showPage()
        c.save()
        print(f"[OK] PDF erzeugt: {os.path.abspath(OUTPUT_PDF)}")
        print(f"[OK] Marker: {MARKER_SIZE_MM}mm, Rand: {EDGE_MARGIN_MM}mm")
        print("Drucken: 100% Skalierung, keine Anpassung an Seite.")
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
