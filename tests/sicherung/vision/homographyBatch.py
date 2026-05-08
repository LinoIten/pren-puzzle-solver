#!/usr/bin/env python3
"""
apply_a4_homography_batch.py

- Berechnet Homographie aus capture_raw.png (mit ArUco-Markern)
- Wendet diese Homographie auf ALLE Bilder in picam_raw/ an
- Speichert die Ergebnisse in picam_raw/ mit Prefix "warp_"
"""

import os
import cv2
import numpy as np



# Einstellungen
REFERENCE_IMAGE = "capture_raw.png"
INPUT_FOLDER = "picam_raw"
OUTPUT_PREFIX = "warp_"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
DPI = 300



# Hilfsfunktionen
def a4_size_pixels(dpi, landscape):
    w_mm, h_mm = 210.0, 297.0
    w_px = int(round((w_mm / 25.4) * dpi))
    h_px = int(round((h_mm / 25.4) * dpi))
    return (h_px, w_px) if landscape else (w_px, h_px)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)



# Homographie bestimmen
ref = cv2.imread(REFERENCE_IMAGE)
if ref is None:
    raise RuntimeError("Referenzbild nicht gefunden")

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

corners, ids, _ = detector.detectMarkers(ref)
if ids is None or len(ids) < 4:
    raise RuntimeError("Nicht genug Marker gefunden")

# 4 größte Marker nehmen
areas = []
for c in corners:
    areas.append(abs(cv2.contourArea(c.reshape(4, 2))))

idx = np.argsort(areas)[::-1][:4]

marker_corners = [corners[i].reshape(4, 2) for i in idx]
marker_centers = [np.mean(c, axis=0) for c in marker_corners]
sheet_center = np.mean(marker_centers, axis=0)

# äußere Ecke pro Marker
outer_points = []
for c in marker_corners:
    dists = np.linalg.norm(c - sheet_center, axis=1)
    outer_points.append(c[np.argmax(dists)])

src = order_points(outer_points)

# Orientierung bestimmen
w = np.linalg.norm(src[1] - src[0])
h = np.linalg.norm(src[3] - src[0])
landscape = w > h

out_w, out_h = a4_size_pixels(DPI, landscape)

dst = np.array([
    [0, 0],
    [out_w - 1, 0],
    [out_w - 1, out_h - 1],
    [0, out_h - 1]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src, dst)

print("Homographie berechnet")


# Batch anwenden
for name in sorted(os.listdir(INPUT_FOLDER)):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    in_path = os.path.join(INPUT_FOLDER, name)
    img = cv2.imread(in_path)
    if img is None:
        continue

    warped = cv2.warpPerspective(img, H, (out_w, out_h))
    out_path = os.path.join(INPUT_FOLDER, OUTPUT_PREFIX + name)
    cv2.imwrite(out_path, warped)

    print("gespeichert:", out_path)

print("Fertig.")
