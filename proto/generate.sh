#!/usr/bin/env bash
#
# Regenerate protobuf bindings from puzzle.proto.
#
# Usage:  ./proto/generate.sh
#
# Outputs:
#   proto/puzzle_pb2.py          — Python bindings
#   proto/puzzle.pb.h            — nanopb C header  (copy to STM32 repo)
#   proto/puzzle.pb.c            — nanopb C source   (copy to STM32 repo)
#
# Requirements: pip install nanopb grpcio-tools protobuf

set -euo pipefail
cd "$(dirname "$0")"

echo "Generating Python protobuf..."
python3 -m grpc_tools.protoc -I. --python_out=. puzzle.proto

echo "Generating nanopb C code..."
python3 -c "
import sys, os
sys.path.insert(0, os.path.dirname('$(python3 -c "import nanopb; print(nanopb.__file__)")'))
" 2>/dev/null || true

NANOPB_GEN=$(python3 -c "import nanopb, os; print(os.path.join(os.path.dirname(nanopb.__file__), 'generator', 'nanopb_generator.py'))")
python3 "$NANOPB_GEN" puzzle.proto

echo ""
echo "Done. Generated files:"
ls -1 puzzle_pb2.py puzzle.pb.h puzzle.pb.c
echo ""
echo "Copy puzzle.pb.h and puzzle.pb.c to the STM32 repo:"
echo "  cp proto/puzzle.pb.h ../PREN2_PuzzleSolver/Core/Inc/communication/"
echo "  cp proto/puzzle.pb.c ../PREN2_PuzzleSolver/Core/Src/communication/"
