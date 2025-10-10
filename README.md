
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
```bash
# Virtuelle Umgebung
python3.11 -m venv venv
source venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt