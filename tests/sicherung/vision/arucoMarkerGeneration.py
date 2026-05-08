# generate_aruco_a4_corners_pdf_and_png.py
#
# Erzeugt:
#   1) eine druckfertige A4-PDF
#   2) zusaetzlich eine PNG-Vorschau mit identischem Layout
#
# WICHTIG beim Drucken:
# - Skalierung: 100%
# - keine Anpassung an Seite
#
# HINWEIS:
# - Die ArUco-Marker werden normal in Schwarz/Weiss erzeugt
# - Danach werden die dunklen Bereiche in Dunkelblau umgefaerbt
# - Fuer die Erkennung sollte das Blau moeglichst dunkel sein

import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import Color

OUTPUT_PDF = "aruco_A4_corners_ids_0_1_2_3.pdf"
OUTPUT_PNG = "aruco_A4_corners_ids_0_1_2_3.png"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]

MARKER_SIZE_MM = 100
EDGE_MARGIN_MM = 5

# Aufloesung der Marker-Bitmap
MARKER_BITMAP_PX = 900

# PNG-Aufloesung fuer Vorschau / Test
PNG_DPI = 300

DRAW_LABELS = True
LABEL_FONT_SIZE = 10


# Farben
# OpenCV benutzt BGR
MARKER_DARK_COLOR = (0, 0, 0)           # echtes Schwarz
# MARKER_DARK_COLOR = (20, 20, 20)        # sehr dunkles Grau
# MARKER_DARK_COLOR = (40, 40, 40)        # dunkles Grau
# MARKER_DARK_COLOR = (0, 60, 0)          # dunkles Gruen in BGR
# MARKER_DARK_COLOR = (0, 80, 0)          # dunkleres Gruen in BGR
#MARKER_DARK_COLOR = (140, 40, 0)          # dunkles Blau in BGR

# MARKER_LIGHT nicht aendern
MARKER_LIGHT_COLOR = (255, 255, 255)      # Weiss in BGR

# ReportLab benutzt RGB im Bereich 0..1
PDF_LABEL_COLOR = Color(
    MARKER_DARK_COLOR[2] / 255.0,  # R
    MARKER_DARK_COLOR[1] / 255.0,  # G
    MARKER_DARK_COLOR[0] / 255.0   # B
)


def getArucoDictName(arucoDict):
    arucoDictNames = {
        cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
        cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
        cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
        cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",

        cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
        cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
        cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
        cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",

        cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
        cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
        cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
        cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",

        cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
        cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
        cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
        cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000",

        cv2.aruco.DICT_ARUCO_ORIGINAL: "DICT_ARUCO_ORIGINAL",
    }

    return arucoDictNames.get(arucoDict, f"unknown aruco dict: {arucoDict}")


def mm_to_px(mm_value, dpi):
    return int(round(mm_value * dpi / 25.4))


def colorize_marker(marker_img, dark_color, light_color=(255, 255, 255)):
    """
    Wandelt einen ArUco-Marker von Graustufe in ein Farbbild um.

    Erwartet:
    - marker_img: 2D uint8 Bild, typischerweise mit 0 fuer dunkel und 255 fuer hell
    - dark_color: z.B. dunkles Blau in BGR
    - light_color: normalerweise Weiss in BGR
    """
    h, w = marker_img.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    color_img[marker_img == 0] = dark_color
    color_img[marker_img == 255] = light_color

    return color_img


def create_pdf():
    page_w_pt, page_h_pt = A4
    page_w_mm = page_w_pt / mm
    page_h_mm = page_h_pt / mm

    needed = 2 * EDGE_MARGIN_MM + MARKER_SIZE_MM
    if needed > page_w_mm or needed > page_h_mm:
        raise ValueError(
            f"Marker+Rand passt nicht auf A4. "
            f"needed={needed:.1f}mm, A4={page_w_mm:.1f}x{page_h_mm:.1f}mm"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    x_left = EDGE_MARGIN_MM
    x_right = page_w_mm - EDGE_MARGIN_MM - MARKER_SIZE_MM
    y_bottom = EDGE_MARGIN_MM
    y_top = page_h_mm - EDGE_MARGIN_MM - MARKER_SIZE_MM

    placements = {
        0: (x_left,  y_top),     # oben links
        1: (x_right, y_top),     # oben rechts
        2: (x_right, y_bottom),  # unten rechts
        3: (x_left,  y_bottom),  # unten links
    }

    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)

    c.setFont("Helvetica", 12)

    pdfInfoText = (
        f"ArUco A4 corners | "
        f"dict: {getArucoDictName(ARUCO_DICT)} | "
        f"marker: {MARKER_SIZE_MM} mm | "
        f"margin: {EDGE_MARGIN_MM} mm | "
        f"print: 100%, no scaling"
    )

    c.drawString(
        EDGE_MARGIN_MM * mm,
        (page_h_mm / 2) * mm,
        pdfInfoText
    )

    tmp_files = []
    try:
        for marker_id in MARKER_IDS:
            marker_img = cv2.aruco.generateImageMarker(
                aruco_dict, marker_id, MARKER_BITMAP_PX
            )

            marker_img_colored = colorize_marker(
                marker_img,
                MARKER_DARK_COLOR,
                MARKER_LIGHT_COLOR
            )

            tmp_name = f"_tmp_aruco_{marker_id}.png"
            cv2.imwrite(tmp_name, marker_img_colored)
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
                c.setFillColor(PDF_LABEL_COLOR)

                label = f"ID {marker_id}"

                if marker_id in (0, 3):  # links
                    lx = (x_mm + MARKER_SIZE_MM + 2) * mm
                else:  # rechts
                    lx = (x_mm - 18) * mm

                if marker_id in (0, 1):  # oben
                    ly = (y_mm + MARKER_SIZE_MM + 2) * mm
                else:  # unten
                    ly = (y_mm - 6) * mm

                c.drawString(lx, ly, label)

                # wieder auf schwarz zuruecksetzen, falls spaeter weiterer Text kommt
                c.setFillColorRGB(0, 0, 0)

        c.showPage()
        c.save()
        print(f"[OK] PDF erzeugt: {os.path.abspath(OUTPUT_PDF)}")

    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


def create_png():
    page_w_mm = 210.0
    page_h_mm = 297.0

    page_w_px = mm_to_px(page_w_mm, PNG_DPI)
    page_h_px = mm_to_px(page_h_mm, PNG_DPI)
    marker_size_px = mm_to_px(MARKER_SIZE_MM, PNG_DPI)
    edge_margin_px = mm_to_px(EDGE_MARGIN_MM, PNG_DPI)

    # Farbbild statt Graustufenbild
    canvas_img = np.full((page_h_px, page_w_px, 3), 255, dtype=np.uint8)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    x_left = edge_margin_px
    x_right = page_w_px - edge_margin_px - marker_size_px
    y_top = edge_margin_px
    y_bottom = page_h_px - edge_margin_px - marker_size_px

    placements = {
        0: (x_left,  y_top),      # oben links
        1: (x_right, y_top),      # oben rechts
        2: (x_right, y_bottom),   # unten rechts
        3: (x_left,  y_bottom),   # unten links
    }

    for marker_id in MARKER_IDS:
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, MARKER_BITMAP_PX
        )

        marker_img = cv2.resize(
            marker_img,
            (marker_size_px, marker_size_px),
            interpolation=cv2.INTER_NEAREST
        )

        marker_img_colored = colorize_marker(
            marker_img,
            MARKER_DARK_COLOR,
            MARKER_LIGHT_COLOR
        )

        x, y = placements[marker_id]
        canvas_img[y:y + marker_size_px, x:x + marker_size_px] = marker_img_colored

        if DRAW_LABELS:
            label = f"ID {marker_id}"

            if marker_id in (0, 3):
                tx = x + marker_size_px + mm_to_px(2, PNG_DPI)
            else:
                tx = max(0, x - mm_to_px(12, PNG_DPI))

            if marker_id in (0, 1):
                ty = max(20, y - mm_to_px(2, PNG_DPI))
            else:
                ty = min(page_h_px - 10, y + marker_size_px + mm_to_px(5, PNG_DPI))

            cv2.putText(
                canvas_img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                MARKER_DARK_COLOR,
                2,
                cv2.LINE_AA
            )

    cv2.imwrite(OUTPUT_PNG, canvas_img)
    print(f"[OK] PNG erzeugt: {os.path.abspath(OUTPUT_PNG)}")
    print(f"[OK] PNG-Groesse: {page_w_px} x {page_h_px} px bei {PNG_DPI} DPI")


def main():
    create_pdf()
    create_png()

    print(f"[OK] ArUco-Dict: {getArucoDictName(ARUCO_DICT)}")
    print(f"[OK] Marker: {MARKER_SIZE_MM} mm, Rand: {EDGE_MARGIN_MM} mm")
    print("Drucken: 100% Skalierung, keine Anpassung an Seite.")
    print(f"[OK] Markerfarbe BGR: {MARKER_DARK_COLOR}")


if __name__ == "__main__":
    main()