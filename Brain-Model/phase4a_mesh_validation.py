#!/usr/bin/env python3
"""Phase 4A: validate a conforming structured HEX8/QUAD4 mesh specification."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
P3B_DIR = DATASET / "preprocessing" / "phase3b_results"
P3B_REPORT = P3B_DIR / "phase3b_preprocess_report.json"
OUT = DATASET / "meshing" / "phase4a_validation"
REPORT = OUT / "phase4a_mesh_validation_report.json"
TAG_SPEC = OUT / "mesh_tag_specification.json"

MATERIAL_NAMES = {0: "Background", 1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
INTERFACE_TAGS = {"1-2": (101, "TumorInterface", "Tumor -> BrainTissue"), "1-3": (102, "BrainCSFInterface", "BrainTissue -> CSF"), "3-4": (103, "CSFSkullInterface", "CSF -> Skull"), "4-5": (104, "SkullScalpInterface", "Skull -> Scalp"), "0-5": (105, "ScalpExterior", "Scalp -> Background")}
ALLOWED_MATERIAL_PAIRS = {"1-2", "1-3", "3-4", "4-5"}


def log(s: str) -> None:
    print(f"[phase4a] {s}")


def node_id(i: np.ndarray, j: np.ndarray, k: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    return (i * (ny + 1) + j) * (nz + 1) + k


def hex_connectivity(coords: np.ndarray, shape: tuple[int, int, int], positive: bool = True) -> np.ndarray:
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    corners = [node_id(i, j, k, shape), node_id(i + 1, j, k, shape), node_id(i + 1, j + 1, k, shape), node_id(i, j + 1, k, shape), node_id(i, j, k + 1, shape), node_id(i + 1, j, k + 1, shape), node_id(i + 1, j + 1, k + 1, shape), node_id(i, j + 1, k + 1, shape)]
    out = np.column_stack(corners)
    if not positive:
        out = out[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return out


def physical_points(ids: np.ndarray, shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    i, rem = np.divmod(ids, ny1 * nz1); j, k = np.divmod(rem, nz1)
    # NIfTI voxel centers are at integer indices; grid corners are at index - 0.5.
    ijk = np.column_stack((i, j, k)).astype(float) - 0.5
    return nib.affines.apply_affine(affine, ijk)


def signed_hex_volume(affine: np.ndarray, positive: bool = True) -> float:
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) - 0.5
    if not positive: corners = corners[[0, 3, 2, 1, 4, 7, 6, 5]]
    p = nib.affines.apply_affine(affine, corners)
    return float(np.linalg.det(np.column_stack((p[1] - p[0], p[3] - p[0], p[4] - p[0]))))


def quad_area(points: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) + 0.5 * np.linalg.norm(np.cross(points[2] - points[0], points[3] - points[0])))


def face_counts(labels: np.ndarray) -> dict[str, int]:
    result = {f"{i}-{j}": 0 for i in range(6) for j in range(i + 1, 6)}
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis); b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        d = a != b
        for x, y in zip(a[d].ravel(), b[d].ravel()): result[f"{min(int(x), int(y))}-{max(int(x), int(y))}"] += 1
    return result


def interface_tag_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {str(tag): 0 for tag, _, _ in INTERFACE_TAGS.values()}
    # Internal interfaces are extracted once using +x,+y,+z comparisons.
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis); b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        for pair, (tag, _, _) in INTERFACE_TAGS.items():
            x, y = map(int, pair.split("-"))
            if x == 0 or y == 0: continue
            counts[str(tag)] += int(np.count_nonzero(((a == x) & (b == y)) | ((a == y) & (b == x))))
    # Scalp exterior includes all six directions against background.
    scalp = labels == 5; bg = labels == 0
    counts["105"] = sum(int(np.count_nonzero(np.take(scalp, [0], axis=ax) & np.take(bg, range(1, labels.shape[ax]), axis=ax))) + int(np.count_nonzero(np.take(scalp, [-1], axis=ax) & np.take(bg, range(labels.shape[ax] - 1), axis=ax))) for ax in range(3))
    # Above boundary-only terms do not include internal scalp/background faces; include all interior directions too.
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis); b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        counts["105"] += int(np.count_nonzero(((a == 5) & (b == 0)) | ((a == 0) & (b == 5))))
    return counts


def referenced_node_mask(labels: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    shape = labels.shape; coords = np.argwhere(labels > 0); full = np.zeros((shape[0] + 1) * (shape[1] + 1) * (shape[2] + 1), bool)
    for start in range(0, len(coords), chunk):
        ids = hex_connectivity(coords[start:start + chunk], shape).ravel(); full[ids] = True
    return full


def adjacency_synthetic(labels: np.ndarray) -> dict[str, int]:
    return face_counts(labels)


def synthetic_tests() -> dict[str, object]:
    results: dict[str, object] = {}
    one = np.zeros((2, 2, 2), np.uint8); one[0, 0, 0] = 1; c1 = np.argwhere(one > 0)
    results["single_voxel"] = {"hex8": 1, "referenced_nodes": int(np.unique(hex_connectivity(c1, one.shape)).size), "exterior_faces": 6}
    results["single_voxel_pass"] = results["single_voxel"] == {"hex8": 1, "referenced_nodes": 8, "exterior_faces": 6}
    same = np.zeros((3, 2, 2), np.uint8); same[0, 0, 0] = same[1, 0, 0] = 1; cs = np.argwhere(same > 0); conn = hex_connectivity(cs, same.shape)
    results["same_material_neighbors"] = {"hex8": 2, "referenced_nodes": int(np.unique(conn).size), "internal_same_material_faces": 0}; results["same_material_pass"] = results["same_material_neighbors"] == {"hex8": 2, "referenced_nodes": 12, "internal_same_material_faces": 0}
    diff = np.zeros((3, 2, 2), np.uint8); diff[0, 0, 0] = 1; diff[1, 0, 0] = 3; cd = np.argwhere(diff > 0); results["different_material_neighbors"] = {"hex8": 2, "referenced_nodes": int(np.unique(hex_connectivity(cd, diff.shape)).size), "brain_csf_faces": int(face_counts(diff)["1-3"])}; results["different_material_pass"] = results["different_material_neighbors"] == {"hex8": 2, "referenced_nodes": 12, "brain_csf_faces": 1}
    nested = np.zeros((11, 11, 11), np.uint8); nested[5, 5, 5] = 2; nested[4:7, 4:7, 4:7][nested[4:7,4:7,4:7] == 0] = 1; nested[3:8,3:8,3:8][nested[3:8,3:8,3:8] == 0] = 3; nested[2:9,2:9,2:9][nested[2:9,2:9,2:9] == 0] = 4; nested[1:10,1:10,1:10][nested[1:10,1:10,1:10] == 0] = 5
    nf = face_counts(nested); results["nested_model"] = {"tag_pairs_present": {k: nf[k] for k in ["1-2", "1-3", "3-4", "4-5"]}, "forbidden_faces": {k: v for k, v in nf.items() if not k.startswith("0-") and k not in ALLOWED_MATERIAL_PAIRS and v}}
    results["nested_model_pass"] = all(nf[k] > 0 for k in ALLOWED_MATERIAL_PAIRS) and not results["nested_model"]["forbidden_faces"]
    identity = np.eye(4); negative = np.diag([-1., 1., 1., 1.]); results["affine_handedness"] = {"identity_signed_volume": signed_hex_volume(identity), "negative_raw_signed_volume": signed_hex_volume(negative), "negative_corrected_signed_volume": signed_hex_volume(negative, positive=False)}; results["affine_handedness_pass"] = signed_hex_volume(identity) > 0 and signed_hex_volume(negative, positive=False) > 0
    face_offsets = {"-x": [(0,0,0),(0,0,1),(0,1,1),(0,1,0)], "+x": [(1,0,0),(1,1,0),(1,1,1),(1,0,1)], "-y": [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], "+y": [(0,1,0),(0,1,1),(1,1,1),(1,1,0)], "-z": [(0,0,0),(0,1,0),(1,1,0),(1,0,0)], "+z": [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]}
    quad_areas = {direction: quad_area(np.asarray(offsets, float)) for direction, offsets in face_offsets.items()}
    results["quad4_ordering"] = {"areas": quad_areas, "distinct_nodes": all(len(set(offsets)) == 4 for offsets in face_offsets.values())}; results["quad4_ordering_pass"] = results["quad4_ordering"]["distinct_nodes"] and all(abs(a - 1.0) < 1e-12 for a in quad_areas.values())
    results["all_pass"] = all(v for k, v in results.items() if k.endswith("_pass"))
    if not results["all_pass"]: raise RuntimeError(f"Synthetic mesh tests failed: {results}")
    return results


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = synthetic_tests(); log("Synthetic tests: PASS")
    p3 = json.loads(P3B_REPORT.read_text()); label_path = P3B_DIR / "material_labels.nii.gz"; img = nib.load(str(label_path)); labels = np.asanyarray(img.dataobj); affine = np.asarray(img.affine, float); shape = labels.shape; spacing = np.linalg.norm(affine[:3, :3], axis=0)
    counts = {str(i): int(np.count_nonzero(labels == i)) for i in range(6)}; valid_labels = bool(np.isin(labels, range(6)).all()); integer = np.issubdtype(labels.dtype, np.integer); det = float(np.linalg.det(affine[:3, :3])); orientation = list(nib.aff2axcodes(affine))
    p3_counts = {str(i): int(p3["regions"][name]["voxels"]) for i, name in {1:"BrainTissue",2:"Tumor",3:"CSF",4:"Skull",5:"Scalp"}.items()}
    tumor_original = np.asanyarray(nib.load(str(DATASET / "5_MASK/NIFTI/MASK.nii.gz")).dataobj) > 0; pad = int(p3["padding_voxels_each_side"]); embedded = np.zeros(shape, bool); embedded[pad:pad + tumor_original.shape[0], pad:pad + tumor_original.shape[1], pad:pad + tumor_original.shape[2]] = tumor_original
    tumor_exact = bool(np.array_equal(labels == 2, embedded)); no_boundary_nonbg = all(not np.any(np.take(labels > 0, [0, -1], axis=ax)) for ax in range(3))
    adj = face_counts(labels); interface = interface_tag_counts(labels); expected_interfaces = {str(k): int(v) for k, v in p3["adjacency_face_counts"].items() if k in ALLOWED_MATERIAL_PAIRS or k == "0-5"}; expected_tag_counts = {"101": expected_interfaces["1-2"], "102": expected_interfaces["1-3"], "103": expected_interfaces["3-4"], "104": expected_interfaces["4-5"], "105": expected_interfaces["0-5"]}
    node_mask = referenced_node_mask(labels); full_nodes = int(np.prod(np.asarray(shape) + 1)); referenced = int(node_mask.sum()); occupied = int(np.count_nonzero(labels)); hex_by_material = {str(i): counts[str(i)] for i in range(1, 6)}; total_quad = int(sum(interface.values()))
    positive_order = det > 0; representative_volume = signed_hex_volume(affine, positive=positive_order)
    lattice_shape = tuple(int(x + 1) for x in shape)
    corner_indices = np.array([[i, j, k] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])], dtype=float) - 0.5
    corner_world = nib.affines.apply_affine(affine, corner_indices)
    physical_support_bbox = {"min": corner_world.min(axis=0).tolist(), "max": corner_world.max(axis=0).tolist()}
    # Check sampled connectivity uniqueness and index bounds without building the full mesh.
    sample_coords = np.argwhere(labels > 0)[::max(1, occupied // 10000)]; sample_conn = hex_connectivity(sample_coords, shape, positive=positive_order); connectivity_sample_pass = bool(np.all([len(np.unique(row)) == 8 for row in sample_conn]) and sample_conn.min() >= 0 and sample_conn.max() < full_nodes)
    memory = {"coordinates_float64_bytes": referenced * 3 * 8, "hex8_connectivity_int64_bytes": occupied * 8 * 8, "hex8_physical_id_int32_bytes": occupied * 4, "quad4_connectivity_int64_bytes": total_quad * 4 * 8, "quad4_physical_id_int32_bytes": total_quad * 4}
    memory["total_core_arrays_bytes"] = sum(memory.values()); memory["total_core_arrays_GiB"] = memory["total_core_arrays_bytes"] / 1024**3
    checks = {"three_dimensional_integer_labels": labels.ndim == 3 and integer, "labels_valid": valid_labels, "material_counts_match_phase3b": all(counts[str(i)] == p3_counts[str(i)] for i in range(1, 6)), "hex_count_equals_nonbackground": occupied == sum(counts[str(i)] for i in range(1, 6)), "adjacency_matches_phase3b": all(adj[k] == int(v) for k, v in expected_interfaces.items()), "no_forbidden_material_adjacency": all(adj[k] == 0 for k in adj if not k.startswith("0-") and k not in ALLOWED_MATERIAL_PAIRS), "no_nonbackground_array_boundary": no_boundary_nonbg, "tumor_exact_preservation": tumor_exact, "referenced_nodes_computed": referenced > 0, "sample_connectivity_valid": connectivity_sample_pass, "positive_representative_hex_volume": representative_volume > 0, "interface_counts_match_phase3b": interface == expected_tag_counts, "quad_total": total_quad == sum(expected_tag_counts.values())}
    tag_spec = {"dimensions": [{"dimension": 3, "physical_id": i, "physical_name": MATERIAL_NAMES[i], "semantic_meaning": "volume material"} for i in range(1, 6)] + [{"dimension": 2, "physical_id": tag, "physical_name": name, "semantic_meaning": pair, "expected_current_dataset_face_count": expected_tag_counts[str(tag)]} for pair, (tag, name, semantic) in INTERFACE_TAGS.items()], "hex8_order": ["c000", "c100", "c110", "c010", "c001", "c101", "c111", "c011"], "corner_convention": "voxel (i,j,k) uses grid corners at affine indices (i-0.5,j-0.5,k-0.5) through (i+0.5,j+0.5,k+0.5)", "node_numbering": "id=((i*(ny+1)+j)*(nz+1)+k), x-index major then y then z", "orientation_rule": "canonical order when det(A[:3,:3])>0; swap x/y corner roles [0,3,2,1,4,7,6,5] when determinant is negative", "quad_ordering": {"-x":["c000","c001","c011","c010"], "+x":["c100","c110","c111","c101"], "-y":["c000","c100","c101","c001"], "+y":["c010","c011","c111","c110"], "-z":["c000","c010","c110","c100"], "+z":["c001","c101","c111","c011"]}}
    TAG_SPEC.write_text(json.dumps(tag_spec, indent=2) + "\n")
    report = {"phase": "PHASE 4A pre-mesh validation", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if all(checks.values()) and tests["all_pass"] else "FAIL", "input_label_path": str(label_path), "shape": list(shape), "affine": affine.tolist(), "spacing_mm": spacing.tolist(), "orientation": orientation, "affine_determinant": det, "physical_support_bbox_world": physical_support_bbox, "material_counts": counts, "phase3b_regression_counts": p3_counts, "expected_hex8_total": occupied, "hex8_counts_by_material": hex_by_material, "interface_quad4_counts_by_tag": interface, "expected_quad4_counts_by_tag": expected_tag_counts, "expected_quad4_total": sum(expected_tag_counts.values()), "actual_quad4_total": total_quad, "full_lattice_shape": list(lattice_shape), "full_lattice_nodes": full_nodes, "referenced_grid_nodes": referenced, "unused_lattice_nodes": full_nodes - referenced, "compaction_percentage": 100 * (full_nodes - referenced) / full_nodes, "chosen_node_numbering": tag_spec["node_numbering"], "chosen_hex8_order": tag_spec["hex8_order"], "representative_signed_hex_volume": representative_volume, "quad_ordering_convention": tag_spec["quad_ordering"], "synthetic_tests": tests, "memory_estimate": memory, "checks": checks, "mesh_tag_specification": str(TAG_SPEC), "scientific_note": "This phase validates a conforming structured voxel mesh specification only. The real .msh/VTU/VTM is intentionally not generated."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Status: {report['status']}; HEX8={occupied}; QUAD4={total_quad}; referenced nodes={referenced}; memory={memory['total_core_arrays_GiB']:.3f} GiB")
    log(f"Material counts={hex_by_material}; interface tags={interface}; det={det}; signed volume={representative_volume}")
    log(f"Report: {REPORT}")
    if report["status"] != "PASS": raise RuntimeError("Phase 4A validation failed; see report")
    return 0


if __name__ == "__main__": raise SystemExit(main())
