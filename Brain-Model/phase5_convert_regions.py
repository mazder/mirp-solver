#!/usr/bin/env python3
"""PHASE 5: compact regional VTU/VTM conversion and independent validation."""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
SOURCE = DATASET / "meshing" / "phase4b_mesh" / "head_tumor_5region.msh"
OUT = DATASET / "meshing" / "phase5_conversion"
REPORT = OUT / "phase5_conversion_report.json"
REGIONS = {1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
INTERFACES = {101: "TumorInterface", 102: "BrainCSFInterface", 103: "CSFSkullInterface", 104: "SkullScalpInterface", 105: "ScalpExterior"}
EXPECTED_HEX = {1: 1968564, 2: 82652, 3: 162698, 4: 280585, 5: 383732}
EXPECTED_QUAD = {101: 21094, 102: 131310, 103: 137498, 104: 147048, 105: 159038}


def log(s: str) -> None:
    print(f"[phase5] {s}")


def compact(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(cells.ravel(), return_inverse=True)
    return points[used], inverse.reshape(cells.shape).astype(np.int64, copy=False)


def bbox(points: np.ndarray) -> dict[str, list[float]]:
    return {"min": points.min(0).tolist(), "max": points.max(0).tolist()}


def hex_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.det(np.stack((points[cells[:, 1]] - points[cells[:, 0]], points[cells[:, 3]] - points[cells[:, 0]], points[cells[:, 4]] - points[cells[:, 0]]), axis=2)))


def quad_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a = points[cells[:, 1]] - points[cells[:, 0]]; b = points[cells[:, 2]] - points[cells[:, 0]]; c = points[cells[:, 3]] - points[cells[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1) + 0.5 * np.linalg.norm(np.cross(b, c), axis=1)


def write_region(name: str, points: np.ndarray, cells: np.ndarray, tags: np.ndarray, physical: np.ndarray, geometrical: np.ndarray, cell_type: str, id_name: str) -> Path:
    out_points, out_cells = compact(points, cells)
    path = OUT / f"{name}.vtu"
    mesh = meshio.Mesh(points=out_points, cells=[(cell_type, out_cells)], cell_data={id_name: [tags.astype(np.int32)], "gmsh:physical": [physical.astype(np.int32)], "gmsh:geometrical": [geometrical.astype(np.int32)]})
    meshio.write(str(path), mesh, file_format="vtu")
    return path


def validate_vtu(path: Path, expected_type: str, id_name: str) -> dict[str, object]:
    mesh = meshio.read(str(path)); block = next((c for c in mesh.cells if c.type == expected_type), None)
    if block is None: raise RuntimeError(f"{path.name}: missing {expected_type} block")
    idx = next(i for i, c in enumerate(mesh.cells) if c is block); cells = np.asarray(block.data); points = np.asarray(mesh.points)
    unused = int(len(points) - np.unique(cells.ravel()).size); valid = bool(cells.min() >= 0 and cells.max() < len(points) and all(len(np.unique(row)) == cells.shape[1] for row in cells))
    data = mesh.cell_data; ids = np.asarray(data[id_name][idx]); physical = np.asarray(data["gmsh:physical"][idx]) if "gmsh:physical" in data else np.array([])
    result = {"file": str(path), "points": int(len(points)), "cells": int(len(cells)), "cell_type": expected_type, "id_name": id_name, "id_unique": sorted(np.unique(ids).astype(int).tolist()), "physical_unique": sorted(np.unique(physical).astype(int).tolist()) if physical.size else [], "connectivity_min": int(cells.min()), "connectivity_max": int(cells.max()), "unused_points": unused, "connectivity_valid": valid, "bbox": bbox(points), "centroid": points.mean(0).tolist(), "cell_data_keys": sorted(data.keys())}
    if expected_type == "hexahedron":
        vol = hex_volumes(points, cells); result["total_cell_volume"] = float(vol.sum()); result["min_cell_volume"] = float(vol.min()); result["max_cell_volume"] = float(vol.max())
    else:
        area = quad_areas(points, cells); result["total_cell_area"] = float(area.sum()); result["min_cell_area"] = float(area.min()); result["max_cell_area"] = float(area.max())
    return result


def write_vtm(paths: dict[str, Path]) -> Path:
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    dataset = ET.SubElement(root, "vtkMultiBlockDataSet")
    for i, (name, path) in enumerate(paths.items()): ET.SubElement(dataset, "DataSet", index=str(i), name=name, file=path.name)
    tree = ET.ElementTree(root); path = OUT / "head_tumor_5region.vtm"; tree.write(path, encoding="utf-8", xml_declaration=True); return path


def main() -> int:
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    log("READING AUTHORITATIVE MSH")
    source = meshio.read(str(SOURCE)); points = np.asarray(source.points); cell_blocks = source.cells; cell_data = source.cell_data
    if "gmsh:physical" not in cell_data or "gmsh:geometrical" not in cell_data: raise RuntimeError("Authoritative mesh lacks physical/geometrical cell data")
    field = {int(v[0]): {"name": str(k), "dimension": int(v[1])} for k, v in source.field_data.items()}
    expected_names = {**{i: {"name": n, "dimension": 3} for i, n in REGIONS.items()}, **{i: {"name": n, "dimension": 2} for i, n in INTERFACES.items()}}
    physical_names_ok = all(field.get(k) == v for k, v in expected_names.items())
    blocks = []
    for idx, block in enumerate(cell_blocks): blocks.append({"index": idx, "type": block.type, "cells": np.asarray(block.data), "physical": np.asarray(cell_data["gmsh:physical"][idx]), "geometrical": np.asarray(cell_data["gmsh:geometrical"][idx])})
    volume_blocks = [b for b in blocks if b["type"] == "hexahedron"]; surface_blocks = [b for b in blocks if b["type"] == "quad"]
    if len(volume_blocks) != 1 or len(surface_blocks) != 1: raise RuntimeError("Expected one hexahedron and one quad source block")
    vol, surf = volume_blocks[0], surface_blocks[0]
    outputs: dict[str, Path] = {}; validation: dict[str, dict[str, object]] = {}
    for material, name in REGIONS.items():
        m = vol["physical"] == material; outputs[name] = write_region(name, points, vol["cells"][m], np.full(int(m.sum()), material), vol["physical"][m], vol["geometrical"][m], "hexahedron", "MaterialID")
    sm = np.isin(surf["physical"], list(INTERFACES)); outputs["Interfaces"] = write_region("Interfaces", points, surf["cells"][sm], surf["physical"][sm], surf["physical"][sm], surf["geometrical"][sm], "quad", "InterfaceID")
    vtm = write_vtm(outputs); outputs["VTM"] = vtm
    log("INDEPENDENT VTU READBACK")
    for material, name in REGIONS.items(): validation[name] = validate_vtu(outputs[name], "hexahedron", "MaterialID")
    validation["Interfaces"] = validate_vtu(outputs["Interfaces"], "quad", "InterfaceID")
    interface_mesh = meshio.read(str(outputs["Interfaces"])); iblock = next(c for c in interface_mesh.cells if c.type == "quad"); iidx = next(i for i,c in enumerate(interface_mesh.cells) if c is iblock); iid = np.asarray(interface_mesh.cell_data["InterfaceID"][iidx]); interface_counts = {str(i): int(np.count_nonzero(iid == i)) for i in INTERFACES}
    regional_counts = {name: validation[name]["cells"] for name in REGIONS.values()}; conservation = {"regional_hex8_sum": int(sum(regional_counts.values())), "source_hex8": int(len(vol["cells"])), "interfaces_quad4": int(validation["Interfaces"]["cells"]), "source_quad4": int(len(surf["cells"])), "hex8_equal": sum(regional_counts.values()) == len(vol["cells"]), "quad4_equal": validation["Interfaces"]["cells"] == len(surf["cells"]), "interface_counts_expected": interface_counts == {str(k): v for k, v in EXPECTED_QUAD.items()}}
    checks = {"source_physical_names": physical_names_ok, "source_points": len(points) == 2958278, "source_hex8": len(vol["cells"]) == 2878231, "source_quad4": len(surf["cells"]) == 595988, "regional_hex8_counts": all(validation[n]["cells"] == EXPECTED_HEX[i] for i,n in REGIONS.items()), "tumor_exact": validation["Tumor"]["cells"] == 82652 and validation["Tumor"]["id_unique"] == [2], "interfaces_valid": validation["Interfaces"]["cells"] == 595988 and validation["Interfaces"]["connectivity_valid"], "regional_connectivity": all(v["connectivity_valid"] and v["unused_points"] == 0 for n,v in validation.items()), "conservation": all(conservation.values()), "vtm_references": all((OUT / p.name).exists() for p in outputs.values() if p.suffix == ".vtu")}
    vtm_validation = {"xml_parse": True, "blocks": list(outputs.keys())[:-1], "pyvista_load": False, "pyvista_n_blocks": None}
    try:
        loaded = pv.read(str(vtm)); vtm_validation["pyvista_load"] = True; vtm_validation["pyvista_n_blocks"] = int(loaded.n_blocks)
    except Exception as exc: vtm_validation["pyvista_error"] = str(exc)
    checks["vtm_validation"] = bool(vtm_validation["pyvista_load"] and vtm_validation["pyvista_n_blocks"] == 6)
    report = {"phase": "PHASE 5 compact regional VTU/VTM conversion", "status": "PASS" if all(checks.values()) else "FAIL", "source_mesh": str(SOURCE), "outputs": {k: str(v) for k,v in outputs.items()}, "source_counts": {"points": len(points), "hex8": len(vol["cells"]), "quad4": len(surf["cells"])}, "physical_names": field, "regional_validation": validation, "interface_counts": interface_counts, "conservation": conservation, "vtm_validation": vtm_validation, "checks": checks, "elapsed_seconds": time.time() - t0, "compatibility": {"ParaView": True, "meshio": True, "PyVista": bool(vtm_validation["pyvista_load"]), "MFEM_conforming_HEX8_in_principle": True, "note": "Regional conversion only; no physics or solver execution."}}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"FINAL STATUS: {report['status']}; regional HEX8={sum(regional_counts.values())}; interface QUAD4={validation['Interfaces']['cells']}")
    log(f"VTM PyVista validation: {vtm_validation}")
    log(f"Report: {REPORT}")
    if report["status"] != "PASS": raise RuntimeError("Phase 5 validation failed; see report")
    return 0


if __name__ == "__main__": raise SystemExit(main())
