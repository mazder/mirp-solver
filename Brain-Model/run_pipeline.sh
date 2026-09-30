#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HDBET_MASK="${ROOT_DIR}/preprocess/masks/brain_extraction_hdbet.nii.gz"
T1_FILE="${ROOT_DIR}/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"

if [[ ! -f "$HDBET_MASK" ]]; then
  command -v hd-bet >/dev/null 2>&1 || {
    echo "[pipeline] ERROR: hd-bet is required to create the anatomical brain mask" >&2
    exit 1
  }
  mkdir -p "${ROOT_DIR}/preprocess/masks"
  echo "[pipeline] HD-BET brain extraction (device=${HDBET_DEVICE:-cpu})"
  hd-bet -i "$T1_FILE" -o "${ROOT_DIR}/preprocess/masks/hd_bet_run" \
    -device "${HDBET_DEVICE:-cpu}" -mode "${HDBET_MODE:-fast}" -tta 0
  test -f "${ROOT_DIR}/preprocess/masks/hd_bet_run_mask.nii.gz"
  mv "${ROOT_DIR}/preprocess/masks/hd_bet_run_mask.nii.gz" "$HDBET_MASK"
fi

echo "[pipeline] preprocessing"
"$PYTHON_BIN" preprocess.py

echo "[pipeline] conforming Gmsh mesh generation"
"$PYTHON_BIN" gmsh_solver.py

echo "[pipeline] regional VTU/VTM extraction"
"$PYTHON_BIN" convert_mesh_mul.py

echo "[pipeline] report generation"
"$PYTHON_BIN" writer.py

echo "[pipeline] independent verification"
"$PYTHON_BIN" verify_pipeline.py

echo "[pipeline] COMPLETE"
