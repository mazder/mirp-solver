#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import meshio
ROOT = Path(__file__).resolve().parent
BASE = ROOT / "EGD-0001" / "MR_EGD-0001" / "results" / "head_tumor_multiregion_mesh"
REGIONS = {1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
def main():
    source = meshio.read(BASE / "head_tumor_5region.msh"); hi = next(i for i,c in enumerate(source.cells) if c.type == "hexahedron"); source_hex = source.cells[hi].data; source_tags = source.cell_data["gmsh:physical"][hi]; expected = {tag: int((source_tags == tag).sum()) for tag in REGIONS}; report = {"source_hex8": len(source_hex), "regions": {}, "pass": True}
    for tag, name in REGIONS.items():
        v = meshio.read(BASE / "multiblock" / f"{name}.vtu"); c = next(x for x in v.cells if x.type == "hexahedron"); ok = len(c.data) == expected[tag] and int(c.data.min()) >= 0 and int(c.data.max()) < len(v.points); report["regions"][name] = {"cells": len(c.data), "expected_source_cells": expected[tag], "points": len(v.points), "connectivity_valid": ok}; report["pass"] &= ok
    i = meshio.read(BASE / "multiblock" / "Interfaces.vtu"); q = next(x for x in i.cells if x.type == "quad"); source_quad = sum(len(c.data) for c in source.cells if c.type == "quad"); report["interfaces_quad4"] = len(q.data); report["expected_source_quad4"] = source_quad; report["pass"] &= len(q.data) == source_quad and (BASE / "multiblock" / "head_tumor_5region.vtm").exists(); print(json.dumps(report, indent=2)); raise SystemExit(0 if report["pass"] else 1)
if __name__ == "__main__": main()
