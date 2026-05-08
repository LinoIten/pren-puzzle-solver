import cv2
import numpy as np

# -----------------------------
# Einstellungen
# -----------------------------
IMAGE_PATH = "aruco_A4_corners_ids_0_1_2_3.png"
OUTPUT_WARP_PATH = "warp_a4.png"
OUTPUT_DEBUG_PATH = "debug_aruco_detection.png"
OUTPUT_H_PATH = "h.npy"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
REQUIRED_IDS = {0, 1, 2, 3}

A4_W_MM = 210.0
A4_H_MM = 297.0

MARKER_SIZE_MM = 45.0
EDGE_MARGIN_MM = 5.0

# Aufloesung des Zielbildes
PX_PER_MM = 10.0
WARP_W = int(round(A4_W_MM * PX_PER_MM))
WARP_H = int(round(A4_H_MM * PX_PER_MM))


def build_marker_world_points_mm_top_left():
    """
    Liefert fuer jede Marker-ID die 4 physischen Eckpunkte in mm
    in einem Zielkoordinatensystem mit Ursprung OBEN LINKS.

    Reihenfolge der Ecken passend zu OpenCV ArUco:
    top-left, top-right, bottom-right, bottom-left
    """

    x_left = EDGE_MARGIN_MM
    x_right = A4_W_MM - EDGE_MARGIN_MM - MARKER_SIZE_MM

    y_top = EDGE_MARGIN_MM
    y_bottom = A4_H_MM - EDGE_MARGIN_MM - MARKER_SIZE_MM

    return {
        # ID 0 = oben links
        0: np.array([
            [x_left, y_top],
            [x_left + MARKER_SIZE_MM, y_top],
            [x_left + MARKER_SIZE_MM, y_top + MARKER_SIZE_MM],
            [x_left, y_top + MARKER_SIZE_MM],
        ], dtype=np.float32),

        # ID 1 = oben rechts
        1: np.array([
            [x_right, y_top],
            [x_right + MARKER_SIZE_MM, y_top],
            [x_right + MARKER_SIZE_MM, y_top + MARKER_SIZE_MM],
            [x_right, y_top + MARKER_SIZE_MM],
        ], dtype=np.float32),

        # ID 2 = unten rechts
        2: np.array([
            [x_right, y_bottom],
            [x_right + MARKER_SIZE_MM, y_bottom],
            [x_right + MARKER_SIZE_MM, y_bottom + MARKER_SIZE_MM],
            [x_right, y_bottom + MARKER_SIZE_MM],
        ], dtype=np.float32),

        # ID 3 = unten links
        3: np.array([
            [x_left, y_bottom],
            [x_left + MARKER_SIZE_MM, y_bottom],
            [x_left + MARKER_SIZE_MM, y_bottom + MARKER_SIZE_MM],
            [x_left, y_bottom + MARKER_SIZE_MM],
        ], dtype=np.float32),
    }


def mm_to_px(points_mm, px_per_mm):
    return points_mm * px_per_mm


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {IMAGE_PATH}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    corners_list, ids, _ = detector.detectMarkers(gray)

    debug_img = image.copy()

    if ids is None or len(ids) == 0:
        raise RuntimeError("Keine ArUco-Marker gefunden.")

    ids = ids.flatten()
    found_ids = set(int(x) for x in ids)

    missing = REQUIRED_IDS - found_ids
    if missing:
        raise RuntimeError(f"Nicht alle benoetigten Marker gefunden. Fehlend: {sorted(missing)}")

    cv2.aruco.drawDetectedMarkers(debug_img, corners_list, ids.reshape(-1, 1))
    cv2.imwrite(OUTPUT_DEBUG_PATH, debug_img)

    world_points_by_id_mm = build_marker_world_points_mm_top_left()

    src_points = []
    dst_points = []

    for marker_corners, marker_id in zip(corners_list, ids):
        marker_id = int(marker_id)

        if marker_id not in REQUIRED_IDS:
            continue

        # OpenCV liefert Form (1,4,2), wir brauchen (4,2)
        img_pts = marker_corners.reshape(4, 2).astype(np.float32)

        # Zielpunkte in mm -> dann in Pixel
        world_pts_mm = world_points_by_id_mm[marker_id]
        world_pts_px = mm_to_px(world_pts_mm, PX_PER_MM).astype(np.float32)

        src_points.extend(img_pts)
        dst_points.extend(world_pts_px)

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    H, mask = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homographie konnte nicht berechnet werden.")

    np.save(OUTPUT_H_PATH, H)

    warped = cv2.warpPerspective(image, H, (WARP_W, WARP_H))
    cv2.imwrite(OUTPUT_WARP_PATH, warped)

    print("[OK] Homographie berechnet.")
    print(f"[OK] H gespeichert in: {OUTPUT_H_PATH}")
    print(f"[OK] Entzerrtes Bild gespeichert in: {OUTPUT_WARP_PATH}")
    print(f"[OK] Debugbild gespeichert in: {OUTPUT_DEBUG_PATH}")
    print(f"[OK] Zielgroesse: {WARP_W} x {WARP_H} px")
    print(f"[OK] Entspricht: {A4_W_MM} x {A4_H_MM} mm bei {PX_PER_MM} px/mm")


if __name__ == "__main__":
    main()