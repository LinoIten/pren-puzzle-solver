# PREN Puzzle Solver
Automatisches Puzzle-Löse-System für PREN HS25/FS26 (HSLU)

## Architektur

### Module
- **vision**: Bildverarbeitung (Kamera, Segmentierung, Features)
- **solver**: Puzzle-Logik (Layout-Solver, Validierung)
- **ui**: User Interface (GUI, Simulator für PREN1)
- **hardware**: Hardware-Steuerung (Motion Control für PREN2)
- **core**: Kern-Pipeline und Konfiguration
- **utils**: Hilfsfunktionen

### Datenfluss
Kamera → Segmentierung → Feature-Extraktion → Solver → Validierung → Hardware

## Setup

### Development Setup
```bash
# Virtuelle Umgebung
python3.13 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Standalone Executable

Build a standalone executable that can run without Python installed:

```bash
# Make build script executable (first time only)
chmod +x build.sh

# Build executable
./build.sh
```

The executable will be created in `dist/PREN-Puzzle-Solver/PREN-Puzzle-Solver`
