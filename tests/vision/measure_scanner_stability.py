#!/usr/bin/env python3
"""
measure_piece_stability_scanner.py

Gleicher Test wie bei der PiCam – nur für Scannerbilder (bereits entzerrt).
Liest scanner_raw/s1.png ... s5.png, segmentiert per Otsu, berechnet:

- Schwerpunkt (cx, cy)
- Flaeche (px^2 und mm^2)
- markiert Schwerpunkt + Kontur + "Ecken" (minAreaRect-Box)
- Flächen-Abweichung relativ zum 1. Bild (s1.png)

Speichert Debug-Bilder im gleichen Ordner als debug_s1.png, debug_s2.png, ...
und druckt eine Auswertung in die Konsole.

Wichtig:
- Fuer korrekte Berechnung der mm^2 braucht das Skript die Scanner-DPI.
"""

import os
import cv2
import numpy as np


INPUT_FOLDER = "scanner_raw"
DEBUG_PREFIX = "debug_"
SCANNER_DPI = 300


def main():
    if not os.path.isdir(INPUT_FOLDER):
        raise RuntimeError(f"Ordner nicht gefunden: {INPUT_FOLDER}")

    # s1.png ... s5.png
    files = [f"s{i}.png" for i in range(1, 6)]

    results = []
    ref_area_mm2 = None

    for name in files:
        path = os.path.join(INPUT_FOLDER, name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("WARN: konnte nicht laden:", path)
            continue

        # mm/px aus DPI
        mm_per_px = 25.4 / float(SCANNER_DPI)   # 1 inch = 25.4 mm
        mm2_per_px2 = mm_per_px * mm_per_px

        #egmentierung: dunkles Teil auf hellem Hintergrund
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu invertiert (Teil -> weiss in Maske)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # cv Voodo-> kleine Bereinigung der Fläche
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Grösste Kontur = Teil
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("WARN: keine Kontur gefunden:", name)
            continue

        cnt = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(cnt))
        area_mm2 = area_px * mm2_per_px2

        # Schwerpunkt berechnen mit Moments
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            print("WARN: m00=0:", name)
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # "Ecken": minAreaRect Box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Referenzflaeche = erstes verarbeitete Bild
        if ref_area_mm2 is None:
            ref_area_mm2 = area_mm2

        abs_dev_mm2 = area_mm2 - ref_area_mm2
        rel_dev_pct = (abs_dev_mm2 / ref_area_mm2) * 100.0 if ref_area_mm2 != 0 else 0.0

        results.append({
            "name": name,
            "cx": cx,
            "cy": cy,
            "area_px": area_px,
            "area_mm2": area_mm2,
            "abs_dev_mm2": abs_dev_mm2,
            "rel_dev_pct": rel_dev_pct,
        })

        # Debug zeichnen
        dbg = img.copy()

        # Kontur
        cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 2)

        # Box (Ecken)
        cv2.polylines(dbg, [box], True, (0, 0, 255), 2)
        for (x, y) in box:
            cv2.circle(dbg, (int(x), int(y)), 6, (0, 0, 255), -1)

        # Schwerpunkt
        cv2.circle(dbg, (int(round(cx)), int(round(cy))), 7, (255, 0, 0), -1)

        # Text
        txt1 = f"area_mm2={area_mm2:.2f}  dev={rel_dev_pct:+.3f}%"
        txt2 = f"centroid=({cx:.1f},{cy:.1f}) px"
        cv2.putText(dbg, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(dbg, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(dbg, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(dbg, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(INPUT_FOLDER, DEBUG_PREFIX + name)
        cv2.imwrite(out_path, dbg)

    if len(results) == 0:
        raise RuntimeError("Keine Ergebnisse (Segmentierung/Konturen fehlgeschlagen).")

    # Ausgabe
    print("\n=== Ergebnisse (Referenz = erstes Bild) ===")
    print(f"Scanner-DPI: {SCANNER_DPI}")
    print(f"Referenz-Flaeche: {ref_area_mm2:.2f} mm^2\n")

    for r in results:
        print(
            f"{r['name']}: "
            f"centroid=({r['cx']:.1f},{r['cy']:.1f}) px, "
            f"area={r['area_mm2']:.2f} mm^2, "
            f"dev={r['rel_dev_pct']:+.3f}% ({r['abs_dev_mm2']:+.2f} mm^2)"
        )

    areas = np.array([r["area_mm2"] for r in results], dtype=np.float64)
    print("\n=== Statistik ===")
    print(f"n={len(areas)}")
    print(f"mean={areas.mean():.2f} mm^2")
    print(f"std ={areas.std(ddof=1) if len(areas) > 1 else 0.0:.4f} mm^2")
    print(f"min ={areas.min():.2f} mm^2")
    print(f"max ={areas.max():.2f} mm^2")
    print(f"range={(areas.max()-areas.min()):.2f} mm^2")


if __name__ == "__main__":
    main()
