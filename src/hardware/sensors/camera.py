import time

# --------------------------------------------------
# TODO: Echte Kamera-Integration implementieren
# --------------------------------------------------
def capture_frame() -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }