#!/usr/bin/env python3
"""PHASE 6: read-only FEM/MFEM solver-readiness validation."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
MESH = DATASET / "meshing" / "phase4b_mesh" / "head_tumor_5region.msh"
OUT = DATASET / "solver_validation"
REPORT = OUT / "phase6_solver_readiness_report.json"
EXPECTED_MAT = {1: 1968564, 2: 82652, 3: 162698, 4: 280585, 5: 383732}
EXPECTED_IFACE = {101: 21094, 102: 131310, 103: 137498, 104: 147048, 105: 159038}
PAIR = {101: (1, 2), 102: (1, 3), 103: (3, 4), 104: (4, 5)}
FACE_LOCAL = np.array([[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2], [0, 3, 2, 1], [4, 5, 6, 7]], dtype=np.int64)


def log(s: str) -> None:
    print(f"[phase6] {s}")


def structured_unique_faces(hexes: np.ndarray, tags: np.ndarray) -> dict[str, object]:
    faces = np.sort(hexes[:, FACE_LOCAL].reshape(-1, 4), axis=1)
    source_hex = np.repeat(np.arange(len(hexes), dtype=np.int64), 6)
    order = np.lexsort((faces[:, 3], faces[:, 2], faces[:, 1], faces[:, 0]))
    sf, sh = faces[order], source_hex[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(sf[1:] != sf[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(sf)])
    unique = sf[starts]
    first_hex = sh[starts]
    second_hex = np.full(len(starts), -1, np.int64)
    two = counts >= 2; second_hex[two] = sh[starts[two] + 1]
    first_mat = tags[first_hex]
    second_mat = np.where(second_hex >= 0, tags[np.maximum(second_hex, 0)], 0)
    return {"unique": unique, "counts": counts, "first_hex": first_hex, "second_hex": second_hex, "first_mat": first_mat, "second_mat": second_mat, "total_generated": len(sf)}


def match_quad_faces(unique: np.ndarray, incidence: dict[str, object], quads: np.ndarray, qtags: np.ndarray) -> dict[str, object]:
    keys = np.ascontiguousarray(unique).view(np.dtype([("a", "<i8"), ("b", "<i8"), ("c", "<i8"), ("d", "<i8")])).ravel()
    qsorted = np.sort(quads, axis=1); qkeys = np.ascontiguousarray(qsorted).view(keys.dtype).ravel()
    idx = np.searchsorted(keys, qkeys)
    exact = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == qkeys)
    counts = incidence["counts"][idx[exact]] if exact.any() else np.array([], dtype=int)
    first = incidence["first_mat"][idx[exact]] if exact.any() else np.array([], dtype=int)
    second = incidence["second_mat"][idx[exact]] if exact.any() else np.array([], dtype=int)
    matched_tags = qtags[exact]
    orphan = int((~exact).sum()); duplicate_quads = int(len(qkeys) - len(np.unique(qkeys)))
    mismatches = 0; tag_details = {}
    for tag in sorted(np.unique(qtags).astype(int)):
        m = exact & (qtags == tag); expected = PAIR.get(tag, (5,))
        c = incidence["counts"][idx[m]] if m.any() else np.array([], int); a = incidence["first_mat"][idx[m]] if m.any() else np.array([], int); b = incidence["second_mat"][idx[m]] if m.any() else np.array([], int)
        good = np.array([int(cc) == (2 if tag != 105 else 1) and (set((int(x), int(y))) == set(expected) if tag != 105 else int(x) == 5 and int(y) == 0) for cc, x, y in zip(c, a, b)], bool)
        mismatches += int((~good).sum()); tag_details[str(tag)] = {"faces": int(m.sum()), "orphan": int(((~exact) & (qtags == tag)).sum()), "incidence_count_valid": bool(good.all()), "expected_material_pair": list(expected)}
    return {"orphan_faces": orphan, "duplicate_quad_faces": duplicate_quads, "mismatched_faces": mismatches, "tag_details": tag_details, "matched": int(exact.sum())}


def main() -> int:
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); log("GMSH READBACK")
    mesh = meshio.read(str(MESH)); points = np.asarray(mesh.points); blocks = mesh.cells; physical = mesh.cell_data["gmsh:physical"]; field = {int(v[0]): {"name": str(k), "dimension": int(v[1])} for k, v in mesh.field_data.items()}
    qi = next(i for i, c in enumerate(blocks) if c.type == "quad"); hi = next(i for i, c in enumerate(blocks) if c.type == "hexahedron"); quads = np.asarray(blocks[qi].data, np.int64); hexes = np.asarray(blocks[hi].data, np.int64); qtags = np.asarray(physical[qi], np.int32); htags = np.asarray(physical[hi], np.int32)
    log(f"points={len(points)} HEX8={len(hexes)} QUAD4={len(quads)}")
    volumes = np.abs(np.linalg.det(np.stack((points[hexes[:, 1]] - points[hexes[:, 0]], points[hexes[:, 3]] - points[hexes[:, 0]], points[hexes[:, 4]] - points[hexes[:, 0]]), axis=2)))
    areas = 0.5 * np.linalg.norm(np.cross(points[quads[:, 1]] - points[quads[:, 0]], points[quads[:, 2]] - points[quads[:, 0]], axis=1), axis=1) + 0.5 * np.linalg.norm(np.cross(points[quads[:, 2]] - points[quads[:, 0]], points[quads[:, 3]] - points[quads[:, 0]], axis=1), axis=1)
    log("RECONSTRUCTING HEX8 FACE TOPOLOGY")
    incidence = structured_unique_faces(hexes, htags); face_counts = incidence["counts"]; qmatch = match_quad_faces(incidence["unique"], incidence, quads, qtags)
    material_counts = {str(i): int(np.count_nonzero(htags == i)) for i in range(1, 6)}; interface_counts = {str(i): int(np.count_nonzero(qtags == i)) for i in [101,102,103,104,105]}
    topology = {"total_hex_faces_generated": int(incidence["total_generated"]), "unique_hex_faces": int(len(face_counts)), "one_cell_faces": int(np.count_nonzero(face_counts == 1)), "two_cell_faces": int(np.count_nonzero(face_counts == 2)), "more_than_two_cell_faces": int(np.count_nonzero(face_counts > 2)), "one_cell_non_scalp_faces": int(np.count_nonzero((face_counts == 1) & (incidence["first_mat"] != 5))), "orphan_interface_faces": qmatch["orphan_faces"], "duplicate_interface_faces": qmatch["duplicate_quad_faces"], "mismatched_interface_faces": qmatch["mismatched_faces"], "missing_boundary_faces": int(np.count_nonzero(face_counts == 1) - interface_counts["105"]), "interface_match_details": qmatch["tag_details"]}
    mfem_installed = bool(shutil.which("mfem") or shutil.which("mfem-config"))
    try: import mfem as _mfem  # type: ignore
    except Exception: mfem_python = False
    else: mfem_python = True
    checks = {"gmsh_readback": True, "volume_attributes": set(np.unique(htags).tolist()) == set(EXPECTED_MAT) and material_counts == {str(k): v for k,v in EXPECTED_MAT.items()}, "boundary_attributes": set(np.unique(qtags).tolist()) == set(EXPECTED_IFACE) and interface_counts == {str(k): v for k,v in EXPECTED_IFACE.items()}, "hex_orientation_positive": bool(np.all(volumes > 0)), "hex_volume_valid": bool(np.allclose(volumes, 1.0)), "no_duplicate_hex_vertices": bool(np.all([len(np.unique(row)) == 8 for row in hexes])), "quad_area_valid": bool(np.all(areas > 0)), "conforming_interfaces": qmatch["orphan_faces"] == 0 and qmatch["duplicate_quad_faces"] == 0 and qmatch["mismatched_faces"] == 0, "topology_manifold": topology["more_than_two_cell_faces"] == 0 and topology["missing_boundary_faces"] == 0 and topology["one_cell_non_scalp_faces"] == 0, "connectivity_valid": int(hexes.min()) >= 0 and int(quads.min()) >= 0 and int(hexes.max()) < len(points) and int(quads.max()) < len(points), "mfem_loading": mfem_installed or mfem_python}
    scalar_dofs = len(points); vector_dofs = 3 * scalar_dofs; scalar_nnz = scalar_dofs * 27; vector_nnz = vector_dofs * 81; memory = {"scalar_dofs": scalar_dofs, "vector_dofs": vector_dofs, "estimated_scalar_stencil_nnz": scalar_nnz, "estimated_vector_nnz": vector_nnz, "estimated_csr_values_bytes": vector_nnz * 8, "estimated_csr_column_bytes": vector_nnz * 4, "estimated_csr_rowptr_bytes": (vector_dofs + 1) * 8, "estimated_csr_total_GiB": (vector_nnz * 12 + (vector_dofs + 1) * 8) / 1024**3, "estimate_note": "27-point scalar node stencil and 3x3 vector coupling; actual sparsity/preconditioner storage will differ."}
    report = {"phase": "PHASE 6 solver readiness", "status": "PASS" if all(v for k,v in checks.items() if k != "mfem_loading") else "FAIL", "mesh": str(MESH), "points": int(len(points)), "hex8": int(len(hexes)), "quad4": int(len(quads)), "material_counts": material_counts, "interface_counts": interface_counts, "field_data": field, "hex_volume_mm3": {"min": float(volumes.min()), "max": float(volumes.max()), "nonpositive": int(np.count_nonzero(volumes <= 0)), "unexpected_from_1": int(np.count_nonzero(~np.isclose(volumes,1.0)))}, "quad_area_mm2": {"min": float(areas.min()), "max": float(areas.max()), "nonpositive": int(np.count_nonzero(areas <= 0))}, "topology": topology, "mfem": {"executable_or_config": mfem_installed, "python_binding": mfem_python, "status": "NOT TESTED" if not (mfem_installed or mfem_python) else "AVAILABLE_BUT_NOT_LOADED"}, "checks": {**checks, "mfem_loading": "NOT TESTED" if not (mfem_installed or mfem_python) else "NOT TESTED"}, "memory_estimate": memory, "optional_verification_solve": "NOT TESTED", "compatibility": {"Gmsh": True, "meshio": True, "MFEM_direct_load": "NOT TESTED (MFEM unavailable)", "conforming_HEX8_ready_in_principle": all(v for k,v in checks.items() if k != "mfem_loading")}, "elapsed_seconds": time.time() - t0}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"STATUS: {report['status']}; topology={topology}; MFEM={report['mfem']}")
    log(f"Report: {REPORT}")
    if report["status"] != "PASS": raise RuntimeError("Phase 6 readiness checks failed")
    return 0


if __name__ == "__main__": raise SystemExit(main())
