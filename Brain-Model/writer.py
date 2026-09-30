#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
RESULT = DATASET / "results" / "head_tumor_multiregion_mesh"
def load(path):
    return json.loads(path.read_text()) if path.exists() else None
def main():
    files = [DATASET / "preprocess" / "preprocess_report.json", RESULT / "mesh_summary.json", RESULT / "mesh_report.json", RESULT / "multiblock" / "multiblock_summary.json"]
    lines = ["MRI-only 5-region FEM pipeline report", "", f"Dataset: {DATASET}", ""]
    for path in files:
        lines += [str(path.relative_to(ROOT)), "-" * 60, json.dumps(load(path), indent=2) if path.exists() else "NOT AVAILABLE", ""]
    out = RESULT / "mesh_summary.txt"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(lines), encoding="utf-8"); print(out)
if __name__ == "__main__": main()
