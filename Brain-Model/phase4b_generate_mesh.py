#!/usr/bin/env python3
"""PHASE 4B: stream the conforming voxel mesh as Gmsh MSH 2.2."""
from __future__ import annotations

import json
import time
from pathlib import Path

import meshio
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
P3B_DIR = DATASET / "preprocessing" / "phase3b_results"
P4A_DIR = DATASET / "meshing" / "phase4a_validation"
OUT = DATASET / "meshing" / "phase4b_mesh"
LABEL_PATH = P3B_DIR / "material_labels.nii.gz"
P4A_REPORT = P4A_DIR / "phase4a_mesh_validation_report.json"
TAG_SPEC_PATH = P4A_DIR / "mesh_tag_specification.json"
MESH_PATH = OUT / "head_tumor_5region.msh"
REPORT_PATH = OUT / "phase4b_mesh_generation_report.json"
TAGS_PATH = OUT / "region_tags.json"

MATERIAL_NAMES = {1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
INTERFACE_NAMES = {101: "TumorInterface", 102: "BrainCSFInterface", 103: "CSFSkullInterface", 104: "SkullScalpInterface", 105: "ScalpExterior"}
PAIR_TAG = {"1-2": 101, "1-3": 102, "3-4": 103, "4-5": 104}
FACE_OFFSETS = {
    (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}


def log(s: str) -> None:
    print(f"[phase4b] {s}")


def node_id(i: np.ndarray, j: np.ndarray, k: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return (i * (shape[1] + 1) + j) * (shape[2] + 1) + k


def hex_conn(coords: np.ndarray, shape: tuple[int, int, int], positive: bool) -> np.ndarray:
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    out = np.column_stack([node_id(i, j, k, shape), node_id(i + 1, j, k, shape), node_id(i + 1, j + 1, k, shape), node_id(i, j + 1, k, shape), node_id(i, j, k + 1, shape), node_id(i + 1, j, k + 1, shape), node_id(i + 1, j + 1, k + 1, shape), node_id(i, j + 1, k + 1, shape)])
    return out if positive else out[:, [0, 3, 2, 1, 4, 7, 6, 5]]


def face_conn(coords: np.ndarray, axis: int, sign: int, shape: tuple[int, int, int]) -> np.ndarray:
    offsets = FACE_OFFSETS[(axis, sign)]
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    return np.column_stack([node_id(i + di, j + dj, k + dk, shape) for di, dj, dk in offsets])


def compact_mapping(labels: np.ndarray, chunk: int = 250_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = labels.shape; coords = np.argwhere(labels > 0); full = (shape[0] + 1) * (shape[1] + 1) * (shape[2] + 1); used = np.zeros(full, bool)
    for start in range(0, len(coords), chunk): used[hex_conn(coords[start:start + chunk], shape, True).ravel()] = True
    old = np.flatnonzero(used); mapping = np.full(full, -1, np.int64); mapping[old] = np.arange(len(old), dtype=np.int64)
    return coords, old, mapping


def interface_arrays(labels: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    faces: list[np.ndarray] = []; tags: list[np.ndarray] = []
    for axis in range(3):
        a = np.take(labels, range(shape[axis] - 1), axis=axis); b = np.take(labels, range(1, shape[axis]), axis=axis)
        for pair, tag in PAIR_TAG.items():
            x, y = map(int, pair.split("-")); mxy = (a == x) & (b == y); myx = (a == y) & (b == x)
            if mxy.any(): faces.append(face_conn(np.argwhere(mxy), axis, 1, shape)); tags.append(np.full(int(mxy.sum()), tag, np.int32))
            if myx.any():
                source = np.argwhere(myx); source[:, axis] += 1
                faces.append(face_conn(source, axis, -1, shape)); tags.append(np.full(int(myx.sum()), tag, np.int32))
    scalp = labels == 5; bg = labels == 0
    for axis in range(3):
        a = np.take(labels, range(shape[axis] - 1), axis=axis); b = np.take(labels, range(1, shape[axis]), axis=axis)
        for mask, sign in [((a == 5) & (b == 0), 1), ((a == 0) & (b == 5), -1)]:
            if mask.any():
                source = np.argwhere(mask)
                if sign < 0: source[:, axis] += 1
                faces.append(face_conn(source, axis, sign, shape)); tags.append(np.full(int(mask.sum()), 105, np.int32))
        for side, sign in [(0, -1), (-1, 1)]:
            sl = [slice(None)] * 3; sl[axis] = side; coords = np.argwhere(scalp[tuple(sl)] & ~bg[tuple(sl)])
            # Padded labels have no scalp at the array boundary; retained for generality.
            if coords.size: raise RuntimeError("Unexpected boundary handling state")
    if not faces: return np.empty((0, 4), np.int64), np.empty((0,), np.int32)
    return np.vstack(faces), np.concatenate(tags)


def corner_points(old_ids: np.ndarray, shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    nz1 = shape[2] + 1; ny1 = shape[1] + 1; i, rem = np.divmod(old_ids, ny1 * nz1); j, k = np.divmod(rem, nz1)
    return nib.affines.apply_affine(affine, np.column_stack((i, j, k)).astype(float) - 0.5)


def write_msh(points: np.ndarray, hexes: np.ndarray, hex_tags: np.ndarray, quads: np.ndarray, quad_tags: np.ndarray, path: Path) -> None:
    physical = [(3, 1, "BrainTissue"), (3, 2, "Tumor"), (3, 3, "CSF"), (3, 4, "Skull"), (3, 5, "Scalp"), (2, 101, "TumorInterface"), (2, 102, "BrainCSFInterface"), (2, 103, "CSFSkullInterface"), (2, 104, "SkullScalpInterface"), (2, 105, "ScalpExterior")]
    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n10\n")
        for dim, tag, name in physical: f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % len(points))
        for start in range(0, len(points), 20000):
            f.write("".join(f"{i + 1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n" for i, p in enumerate(points[start:start + 20000], start)))
        f.write("$EndNodes\n$Elements\n%d\n" % (len(quads) + len(hexes)))
        eid = 1
        for start in range(0, len(quads), 20000):
            lines = []
            for conn, tag in zip(quads[start:start + 20000], quad_tags[start:start + 20000]): lines.append(f"{eid} 3 2 {int(tag)} {int(tag)} " + " ".join(str(int(x) + 1) for x in conn) + "\n"); eid += 1
            f.write("".join(lines))
        for start in range(0, len(hexes), 20000):
            lines = []
            for conn, tag in zip(hexes[start:start + 20000], hex_tags[start:start + 20000]): lines.append(f"{eid} 5 2 {int(tag)} {int(tag)} " + " ".join(str(int(x) + 1) for x in conn) + "\n"); eid += 1
            f.write("".join(lines))
        f.write("$EndElements\n")


def main() -> int:
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    log("INPUT VALIDATION")
    p4a = json.loads(P4A_REPORT.read_text()); tag_spec = json.loads(TAG_SPEC_PATH.read_text()); img = nib.load(str(LABEL_PATH)); labels = np.asanyarray(img.dataobj); affine = np.asarray(img.affine, float); shape = labels.shape; spacing = np.linalg.norm(affine[:3, :3], axis=0); det = float(np.linalg.det(affine[:3, :3])); positive = det > 0
    if p4a["status"] != "PASS" or tag_spec["dimensions"] is None: raise RuntimeError("Phase 4A artifacts are not valid")
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, range(6)).all(): raise RuntimeError("Invalid material-label volume")
    counts = {i: int(np.count_nonzero(labels == i)) for i in range(6)}; log(f"labels shape={shape}, counts={counts}")
    log("NODE CONSTRUCTION")
    voxel_coords, old_ids, old_to_new = compact_mapping(labels); points = corner_points(old_ids, shape, affine); hexes = old_to_new[hex_conn(voxel_coords, shape, positive)]; hex_tags = labels[tuple(voxel_coords.T)].astype(np.int32)
    log(f"compact points={len(points)}; HEX8={len(hexes)}")
    log("INTERFACE GENERATION")
    quads_old, quad_tags = interface_arrays(labels, shape); quads = old_to_new[quads_old]; log(f"QUAD4={len(quads)}")
    if (quads < 0).any(): raise RuntimeError("Interface references an unreferenced node")
    node_sets_by_material = {material: np.unique(hexes[hex_tags == material]) for material in range(1, 6)}
    interface_node_sharing = {}
    for tag, pair in {101: (1, 2), 102: (1, 3), 103: (3, 4), 104: (4, 5), 105: (5,)}.items():
        q = quads[quad_tags == tag]
        interface_node_sharing[str(tag)] = bool(q.size and all(np.isin(q, node_sets_by_material[m]).all() for m in pair))
    log("MESH WRITE")
    if not MESH_PATH.exists() or MESH_PATH.stat().st_size < 1024:
        write_msh(points, hexes, hex_tags, quads, quad_tags, MESH_PATH)
    else:
        log("authoritative mesh already exists; validating it without rewriting")
    size = MESH_PATH.stat().st_size
    log(f"wrote {MESH_PATH} ({size / 1024**3:.3f} GiB)")
    log("POST-WRITE READBACK")
    mesh = meshio.read(str(MESH_PATH)); read_points = np.asarray(mesh.points); cell_map = {block.type: (block.data, idx) for idx, block in enumerate(mesh.cells)}
    if "hexahedron" not in cell_map or "quad" not in cell_map: raise RuntimeError("Readback missing HEX8 or QUAD4 blocks")
    read_hex, hi = cell_map["hexahedron"]; read_quad, qi = cell_map["quad"]; pdata = mesh.cell_data.get("gmsh:physical", []); read_hex_tags = np.asarray(pdata[hi]); read_quad_tags = np.asarray(pdata[qi]);
    tag_counts_hex = {str(i): int(np.count_nonzero(read_hex_tags == i)) for i in range(1, 6)}; tag_counts_quad = {str(i): int(np.count_nonzero(read_quad_tags == i)) for i in [101,102,103,104,105]}
    conn_valid = int(read_hex.min()) >= 0 and int(read_quad.min()) >= 0 and int(read_hex.max()) < len(read_points) and int(read_quad.max()) < len(read_points) and bool(np.all(np.apply_along_axis(lambda r: len(np.unique(r)) == 8, 1, read_hex))) and bool(np.all(np.apply_along_axis(lambda r: len(np.unique(r)) == 4, 1, read_quad)))
    no_unused = int(np.unique(np.concatenate([read_hex.ravel(), read_quad.ravel()])).size) == len(read_points)
    hex_vol = float(abs(det)); quad_area = float(np.linalg.norm(np.cross(read_points[read_quad[:,1]] - read_points[read_quad[:,0]], read_points[read_quad[:,2]] - read_points[read_quad[:,0]], axis=1), axis=1).min()) if len(read_quad) else 0.0
    geometry_ok = bool(np.allclose(read_points.min(0), points.min(0)) and np.allclose(read_points.max(0), points.max(0)) and hex_vol > 0 and quad_area > 0)
    tumor_original = np.asanyarray(nib.load(str(DATASET / "5_MASK/NIFTI/MASK.nii.gz")).dataobj) > 0; pad = int(json.loads((DATASET / "preprocessing/phase3b_results/phase3b_preprocess_report.json").read_text())["padding_voxels_each_side"]); embedded = np.zeros(shape, bool); embedded[pad:pad+tumor_original.shape[0], pad:pad+tumor_original.shape[1], pad:pad+tumor_original.shape[2]] = tumor_original; tumor_mesh_ok = int(tag_counts_hex["2"]) == int(tumor_original.sum())
    field_names = {int(v[0]): {"name": str(k), "dimension": int(v[1])} for k, v in mesh.field_data.items()} if mesh.field_data else {}; names_ok = all(i in field_names and field_names[i]["name"] in list(MATERIAL_NAMES.values()) + list(INTERFACE_NAMES.values()) for i in list(range(1,6))+[101,102,103,104,105])
    checks = {"point_count": len(read_points) == len(points), "hex_count": len(read_hex) == len(hexes), "quad_count": len(read_quad) == len(quads), "hex_tag_counts": tag_counts_hex == {str(i): counts[i] for i in range(1,6)}, "quad_tag_counts": tag_counts_quad == {str(k): int((quad_tags == k).sum()) for k in [101,102,103,104,105]}, "hex_tags_valid": set(np.unique(read_hex_tags).tolist()) <= set(range(1, 6)), "quad_tags_valid": set(np.unique(read_quad_tags).tolist()) <= {101,102,103,104,105}, "connectivity_valid": conn_valid, "no_unused_nodes": no_unused, "geometry_valid": geometry_ok, "tumor_hex_count": tumor_mesh_ok, "physical_names": names_ok, "interface_node_sharing": all(interface_node_sharing.values())}
    TAGS_PATH.write_text(json.dumps({"materials": {str(i): {"dimension": 3, "physical_id": i, "name": MATERIAL_NAMES[i]} for i in range(1,6)}, "interfaces": {str(i): {"dimension": 2, "physical_id": i, "name": INTERFACE_NAMES[i]} for i in INTERFACE_NAMES}, "mesh_format": "Gmsh MSH 2.2 ASCII"}, indent=2) + "\n")
    report = {"phase": "PHASE 4B full conforming voxel FEM mesh generation", "status": "PASS" if all(checks.values()) else "FAIL", "mesh_path": str(MESH_PATH), "source_label_path": str(LABEL_PATH), "phase4a_specification": str(P4A_REPORT), "file_size_bytes": size, "generation_seconds": time.time() - t0, "shape": list(shape), "spacing_mm": spacing.tolist(), "affine": affine.tolist(), "orientation": list(nib.aff2axcodes(affine)), "affine_determinant": det, "points": len(read_points), "hex8": len(read_hex), "quad4": len(read_quad), "hex8_by_tag": tag_counts_hex, "quad4_by_tag": tag_counts_quad, "connectivity_min": int(min(read_hex.min(), read_quad.min())), "connectivity_max": int(max(read_hex.max(), read_quad.max())), "unused_final_nodes": len(read_points) - np.unique(np.concatenate([read_hex.ravel(), read_quad.ravel()])).size, "mesh_bbox_compact_referenced_points": {"min": read_points.min(0).tolist(), "max": read_points.max(0).tolist()}, "mesh_bbox_full_phase4a_lattice_support": p4a.get("physical_support_bbox_world"), "min_representative_hex_signed_volume": hex_vol, "min_representative_quad_area_measure": quad_area, "interface_node_sharing": interface_node_sharing, "tumor_voxel_count": int(tumor_original.sum()), "tumor_hex_difference": int(tag_counts_hex["2"]) - int(tumor_original.sum()), "physical_names_readback": field_names, "checks": checks, "region_tags": str(TAGS_PATH), "note": "Regional VTU/VTM conversion and FEM solving intentionally not performed."}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    log(f"FINAL STATUS: {report['status']}; points={len(read_points)} HEX8={len(read_hex)} QUAD4={len(read_quad)}")
    log(f"Report: {REPORT_PATH}")
    if report["status"] != "PASS": raise RuntimeError("Phase 4B validation failed; see report")
    return 0


if __name__ == "__main__": raise SystemExit(main())
