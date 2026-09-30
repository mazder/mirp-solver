#!/usr/bin/env python3
"""Read-only extraction of tagged HEX8/QUAD4 regions from the Gmsh mesh."""
from __future__ import annotations
import json
from pathlib import Path
import meshio
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
MESH_DIR = DATASET / "results" / "head_tumor_multiregion_mesh"
SOURCE = MESH_DIR / "head_tumor_5region.msh"
OUT = MESH_DIR / "multiblock"
VOLUMES = {1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
SURFACES = {101: "TumorInterface", 102: "BrainCSFInterface", 103: "CSFSkullInterface", 104: "SkullScalpInterface", 105: "ScalpExterior"}

def compact(points, cells):
    used = np.unique(cells.reshape(-1)).astype(np.int64)
    if len(used) and (used[0] < 0 or used[-1] >= len(points)): raise RuntimeError("Invalid source point index")
    remap = np.full(len(points), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    return points[used].copy(), remap[cells], used

def tags_for(mesh, index):
    arrays = mesh.cell_data.get("gmsh:physical")
    if arrays is None or index >= len(arrays): raise RuntimeError("Missing gmsh:physical data")
    tags = np.asarray(arrays[index])
    if len(tags) != len(mesh.cells[index].data): raise RuntimeError(f"Cell block {index} and tags are misaligned")
    return tags

def main():
    if not SOURCE.exists(): raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = meshio.read(SOURCE)
    expected = {**{t: (n, 3) for t, n in VOLUMES.items()}, **{t: (n, 2) for t, n in SURFACES.items()}}
    actual = {int(v[0]): (name, int(v[1])) for name, v in mesh.field_data.items()}
    if actual != expected: raise RuntimeError(f"Physical names/tags mismatch: {actual}")
    hex_blocks = [i for i,c in enumerate(mesh.cells) if c.type == "hexahedron"]
    quad_blocks = [i for i,c in enumerate(mesh.cells) if c.type == "quad"]
    if not hex_blocks or not quad_blocks: raise RuntimeError("Expected HEX8 and QUAD4 blocks")
    points = np.asarray(mesh.points, dtype=float); results = {}; total_hex = 0
    expected_counts = None
    report_path = MESH_DIR / "mesh_summary.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        expected_counts = {int(k): int(v) for k, v in {
            str(tag): report.get("region_voxels", {}).get(name, -1)
            for tag, name in VOLUMES.items()}.items()}
    for tag, name in VOLUMES.items():
        parts = [mesh.cells[i].data[tags_for(mesh, i) == tag] for i in hex_blocks]
        cells = np.concatenate(parts) if parts else np.empty((0, 8), dtype=np.int64)
        if expected_counts is not None and len(cells) != expected_counts[tag]:
            raise RuntimeError(f"{name}: {len(cells)} HEX8, expected preprocessing count {expected_counts[tag]}")
        p, c, used = compact(points, np.asarray(cells, dtype=np.int64)); out = OUT / f"{name}.vtu"
        meshio.write(out, meshio.Mesh(p, [("hexahedron", c)], cell_data={"MaterialID": [np.full(len(c), tag, dtype=np.int32)], "PhysicalID": [np.full(len(c), tag, dtype=np.int32)]}, field_data={name: np.array([tag, 3], dtype=np.int32)}))
        total_hex += len(c)
        results[name] = {"physical_id": tag, "hex8_cells": len(c), "points": len(p), "bbox_min": p.min(0).tolist(), "bbox_max": p.max(0).tolist(), "connectivity_min": int(c.min()), "connectivity_max": int(c.max()), "unique_referenced_points": int(len(used)), "file": out.name}
    qparts, qtags = [], []
    for i in quad_blocks:
        t = tags_for(mesh, i)
        for tag in SURFACES:
            qparts.append(mesh.cells[i].data[t == tag]); qtags.append(np.full(np.count_nonzero(t == tag), tag, dtype=np.int32))
    qcells = np.concatenate([x for x in qparts if len(x)]); qids = np.concatenate([x for x in qtags if len(x)])
    source_quad_total = sum(len(mesh.cells[i].data) for i in quad_blocks)
    if len(qcells) != source_quad_total: raise RuntimeError(f"QUAD4 count {len(qcells)} != source count {source_quad_total}")
    p, c, used = compact(points, qcells.astype(np.int64)); meshio.write(OUT / "Interfaces.vtu", meshio.Mesh(p, [("quad", c)], cell_data={"PhysicalID": [qids], "InterfaceID": [qids]}))
    results["Interfaces"] = {"quad4_cells": len(c), "points": len(p), "physical_regions": {SURFACES[t]: int(np.count_nonzero(qids == t)) for t in SURFACES}, "bbox_min": p.min(0).tolist(), "bbox_max": p.max(0).tolist(), "file": "Interfaces.vtu"}
    names = ["BrainTissue", "Tumor", "CSF", "Skull", "Scalp", "Interfaces"]
    xml = ['<?xml version="1.0"?>', '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">', '<vtkMultiBlockDataSet>'] + [f'<DataSet index="{i}" name="{n}" file="{n}.vtu"/>' for i,n in enumerate(names)] + ['</vtkMultiBlockDataSet>', '</VTKFile>']
    (OUT / "head_tumor_5region.vtm").write_text("\n".join(xml) + "\n", encoding="utf-8")
    source_hex_total = sum(len(mesh.cells[i].data) for i in hex_blocks)
    if total_hex != source_hex_total or len(qcells) != source_quad_total:
        raise RuntimeError("Element conservation against source cell blocks failed")
    summary = {"source_mesh": str(SOURCE), "source_points": int(len(points)), "source_cell_blocks": [{"type": c.type, "count": len(c.data)} for c in mesh.cells], "regions": results, "total_hex8": total_hex, "expected_total_hex8": source_hex_total, "total_quad4": int(len(qcells)), "expected_total_quad4": source_quad_total, "hex8_conservation": total_hex == source_hex_total, "quad4_conservation": len(qcells) == source_quad_total, "read_only_source": True}
    (OUT / "multiblock_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
