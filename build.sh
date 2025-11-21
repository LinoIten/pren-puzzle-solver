#!/bin/bash

source venv/bin/activate
pip install pyinstaller

# Clean
rm -rf build dist

# Build with spec file
pyinstaller --clean PREN-Puzzle-Solver.spec

echo ""
echo "✓ Build complete!"
echo "Test: ./dist/PREN-Puzzle-Solver/PREN-Puzzle-Solver"