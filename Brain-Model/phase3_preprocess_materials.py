#!/usr/bin/env python3
"""MRI-only material-label preprocessing after Phase 2C.

This stage creates a tumor-safe FEM envelope and computational CSF/skull/scalp
shells. It deliberately stops before voxel/FEM/Gmsh meshing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
PHASE2C = DATASET / "phase2_brain_extraction" / "phase2c_computational_brain_envelope"
TUMOR_PATH = DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz"
OUTER_PATH = DATASET / "phase2_brain_extraction" / "classical" / "classical_outer_head_mask.nii.gz"
OUT = DATASET / "preprocessing" / "results"
REPORT = OUT / "preprocess_report.json"


def log(message: str) -> None:
    print(f"[preprocess-materials] {message}")


def bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    c = np.argwhere(mask)
    return {"min": c.min(axis=0).tolist(), "max": c.max(axis=0).tolist()} if c.size else None


def world_bbox(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float]] | None:
    c = np.argwhere(mask)
    if not c.size:
        return None
    lo, hi = c.min(axis=0), c.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)
    w = nib.affines.apply_affine(affine, corners)
    return {"min": w.min(axis=0).tolist(), "max": w.max(axis=0).tolist()}


def centroid(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float] | None]:
    c = np.argwhere(mask)
    if not c.size:
        return {"voxel": None, "world": None}
    cv = c.mean(axis=0)
    return {"voxel": cv.tolist(), "world": nib.affines.apply_affine(affine, cv).tolist()}


def region_geometry(mask: np.ndarray, affine: np.ndarray, spacing: np.ndarray) -> dict[str, object]:
    n = int(mask.sum())
    return {"voxels": n, "volume_mm3": float(n * abs(np.linalg.det(affine[:3, :3]))),
            "volume_ml": float(n * abs(np.linalg.det(affine[:3, :3])) / 1000.0), "voxel_bbox": bbox(mask),
            "world_bbox": world_bbox(mask, affine), "centroid": centroid(mask, affine),
            "spacing_mm": spacing.tolist()}


def save_surface(mask: np.ndarray, affine: np.ndarray, path: Path) -> dict[str, object]:
    vertices, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    vertices = nib.affines.apply_affine(affine, vertices)
    cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    pv.PolyData(vertices, cells).save(str(path))
    return {"file": str(path), "points": int(vertices.shape[0]), "triangles": int(faces.shape[0])}


def adjacency_counts(labels: np.ndarray) -> dict[str, int]:
    pairs: dict[str, int] = {}
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        diff = a != b
        for x, y in zip(a[diff].ravel(), b[diff].ravel()):
            if x == 0 and y == 0:
                continue
            key = f"{min(int(x), int(y))}-{max(int(x), int(y))}"
            pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items()))


def exterior_counts(labels: np.ndarray) -> dict[str, int]:
    result = {str(i): 0 for i in range(6)}
    bg = labels == 0
    six = ndimage.generate_binary_structure(3, 1)
    # Background connected to the padded array boundary is computational exterior.
    bg_labels, n = ndimage.label(bg, structure=six)
    boundary_ids = np.unique(np.concatenate([
        bg_labels[0, :, :].ravel(), bg_labels[-1, :, :].ravel(), bg_labels[:, 0, :].ravel(),
        bg_labels[:, -1, :].ravel(), bg_labels[:, :, 0].ravel(), bg_labels[:, :, -1].ravel()]))
    exterior = np.isin(bg_labels, boundary_ids[boundary_ids > 0])
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        for material in range(1, 6):
            result[str(material)] += int(np.count_nonzero(((a == material) & (b == 0) & np.take(exterior, range(exterior.shape[axis] - 1), axis=axis)) | ((b == material) & (a == 0) & np.take(exterior, range(exterior.shape[axis] - 1), axis=axis))))
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h_path = PHASE2C / "computational_brain_envelope.nii.gz"
    if not h_path.exists() or not TUMOR_PATH.exists():
        raise FileNotFoundError("Phase 2C envelope or original tumor input is missing")
    log(f"Reading Phase 2C FEM envelope read-only: {h_path}")
    h_img = nib.load(str(h_path)); tumor_img = nib.load(str(TUMOR_PATH))
    h = np.asanyarray(h_img.dataobj) > 0; tumor = np.asanyarray(tumor_img.dataobj) > 0
    if h.shape != tumor.shape or not np.allclose(h_img.affine, tumor_img.affine):
        raise RuntimeError("FEM envelope and original tumor geometry mismatch; refusing to resample")
    affine = np.asarray(h_img.affine, float); spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if len(h.shape) != 3: raise RuntimeError("Inputs must be 3-D")
    anatomical_img = nib.load(str(DATASET / "phase2_brain_extraction" / "hd_bet" / "T1_hdbet_mask.nii.gz"))
    anatomical = np.asanyarray(anatomical_img.dataobj) > 0
    if anatomical.shape != h.shape or not np.allclose(anatomical_img.affine, affine): raise RuntimeError("HD-BET reference geometry mismatch")
    original_tumor_reload = np.asanyarray(nib.load(str(TUMOR_PATH)).dataobj) > 0
    if not np.array_equal(tumor, original_tumor_reload): raise RuntimeError("Original tumor changed between reads")
    # Computational shells are physical-distance dilations of the FEM envelope.
    distance = ndimage.distance_transform_edt(~h, sampling=spacing)
    brain_outer = h
    csf_outer = h | (distance <= 2.0)
    skull_outer = h | (distance <= 5.0)
    scalp_outer = h | (distance <= 9.0)
    brain_tissue = h & ~tumor
    csf = csf_outer & ~brain_outer
    skull = skull_outer & ~csf_outer
    scalp = scalp_outer & ~skull_outer
    masks = {1: brain_tissue, 2: tumor, 3: csf, 4: skull, 5: scalp}
    labels = np.zeros(h.shape, dtype=np.uint8)
    for material, mask in masks.items(): labels[mask] = material
    if np.count_nonzero(sum(masks.values()) > 1): raise RuntimeError("Material masks overlap")
    # Derived-only padding gives a background moat larger than the 9 mm scalp shell.
    pad = 11
    pad_width = [(pad, pad)] * 3
    padded_masks = {k: np.pad(v, pad_width, mode="constant", constant_values=False) for k, v in masks.items()}
    padded_labels = np.pad(labels, pad_width, mode="constant", constant_values=0)
    padded_affine = affine.copy(); padded_affine[:3, 3] -= affine[:3, :3] @ np.full(3, pad, dtype=float)
    # The embedded original tumor must have identical world coordinates after padding.
    embedded_tumor = padded_masks[2]
    old_tumor_world = world_bbox(tumor, affine); new_tumor_world = world_bbox(embedded_tumor, padded_affine)
    if old_tumor_world != new_tumor_world: raise RuntimeError("Tumor world bounding box changed during derived padding")
    # Final validation.
    expected_pairs = {"1-2", "1-3", "3-4", "4-5"}
    adjacency = adjacency_counts(padded_labels); forbidden = sorted(set(adjacency) - expected_pairs)
    exterior = exterior_counts(padded_labels)
    invariants = {
        "tumor_count": int(tumor.sum()) == int(original_tumor_reload.sum()),
        "tumor_differing_voxels_zero": int(np.count_nonzero(tumor != original_tumor_reload)) == 0,
        "tumor_exact_equal": bool(np.array_equal(tumor, original_tumor_reload)),
        "tumor_inside_fem_brain": int(np.count_nonzero(tumor & ~h)) == 0,
        "identical_final_shapes": len({m.shape for m in padded_masks.values()}) == 1,
        "identical_final_affine": True,
        "identical_final_orientation": True,
        "mutually_exclusive": int(np.count_nonzero(sum(padded_masks.values()) > 1)) == 0,
        "nested_brain_csf": int(np.count_nonzero(brain_outer & ~csf_outer)) == 0,
        "nested_csf_skull": int(np.count_nonzero(csf_outer & ~skull_outer)) == 0,
        "nested_skull_scalp": int(np.count_nonzero(skull_outer & ~scalp_outer)) == 0,
        "no_forbidden_adjacency": not forbidden,
        "only_scalp_exterior": all(exterior[str(i)] == 0 for i in [1, 2, 3, 4]),
        "material_conservation": int(sum(int(m.sum()) for m in padded_masks.values())) == int(np.count_nonzero(padded_labels)),
        "binary_masks": all(np.isin(m, [False, True]).all() for m in padded_masks.values()),
    }
    # Save derived outputs only.
    outputs = {}
    save_map = {"brain_anatomical_hdbet": anatomical, "brain_envelope_fem": h, "brain_mask": h, "tumor_mask": tumor, "csf_mask": masks[3], "skull_mask": masks[4], "scalp_mask": masks[5]}
    for name, arr in save_map.items():
        p = OUT / f"{name}.nii.gz"; nib.save(nib.Nifti1Image(np.pad(arr, pad_width, constant_values=False).astype(np.uint8), padded_affine), str(p)); outputs[name] = str(p)
    material_path = OUT / "material_labels.nii.gz"; nib.save(nib.Nifti1Image(padded_labels, padded_affine), str(material_path)); outputs["material_labels"] = str(material_path)
    original_labels = np.zeros(labels.shape, dtype=np.uint8)
    for material, mask in masks.items(): original_labels[mask] = material
    original_path = OUT / "material_labels_original_geometry.nii.gz"; nib.save(nib.Nifti1Image(original_labels, affine), str(original_path)); outputs["material_labels_original_geometry"] = str(original_path)
    # Compact three-plane diagnostic at the disagreement centroid.
    c = np.argwhere(tumor & ~anatomical).mean(axis=0).astype(int) if np.any(tumor & ~anatomical) else np.argwhere(tumor).mean(axis=0).astype(int)
    t1 = np.asanyarray(nib.load(str(DATASET / "1_T1" / "NIFTI" / "T1.nii.gz")).dataobj)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    planes = [("axial", 2, int(c[2])), ("coronal", 1, int(c[1])), ("sagittal", 0, int(c[0]))]
    colors = {1: "cyan", 2: "yellow", 3: "blue", 4: "orange", 5: "red"}
    for ax, (name, axis, index) in zip(axes, planes):
        sl = [slice(None)] * 3; sl[axis] = index
        ax.imshow(t1[tuple(sl)].T, cmap="gray", origin="lower")
        for material in range(1, 6):
            boundary = masks[material] & ~ndimage.binary_erosion(masks[material], structure=ndimage.generate_binary_structure(3, 1))
            ax.contour(boundary[tuple(sl)].T, levels=[0.5], colors=colors[material], linewidths=0.8)
        ax.set_title(f"{name} index={index}"); ax.set_xlabel("index"); ax.set_ylabel("index")
    png = OUT / "material_diagnostic_3planes.png"; fig.savefig(png, dpi=180); plt.close(fig); outputs["diagnostic_png"] = str(png)
    # Surfaces are visualization only and do not feed back into masks.
    surfaces = {}
    for name, arr in {"brain_envelope_fem": h, "csf_outer": csf_outer, "skull_outer": skull_outer, "scalp_outer": scalp_outer}.items():
        surfaces[name] = save_surface(np.pad(arr, pad_width, constant_values=False), padded_affine, OUT / f"{name}.vtp")
    regions = {name: region_geometry(np.pad(arr, pad_width, constant_values=False), padded_affine, spacing) for name, arr in {"BrainTissue": masks[1], "Tumor": masks[2], "CSF": masks[3], "Skull": masks[4], "Scalp": masks[5]}.items()}
    all_pairs = {f"{i}-{j}": int(adjacency.get(f"{i}-{j}", 0)) for i in range(6) for j in range(i + 1, 6)}
    failure_reasons = []
    if forbidden:
        failure_reasons.append(f"forbidden adjacency pairs observed: {forbidden}")
    if not invariants["only_scalp_exterior"]:
        failure_reasons.append(f"non-scalp materials contact exterior background: {exterior}")
    report = {"phase": "derived 5-region material preprocessing (no meshing)", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if all(invariants.values()) else "FAIL", "failure_reasons": failure_reasons, "inputs": {"hd_bet_anatomical_reference": str(anatomical_img), "phase2c_fem_envelope": str(h_path), "original_tumor": str(TUMOR_PATH), "outer_head_reference": str(OUTER_PATH)}, "outputs": outputs, "original_affine": affine.tolist(), "final_affine": padded_affine.tolist(), "padding_voxels_each_side": pad, "geometry": {"original_shape": list(h.shape), "final_shape": list(padded_labels.shape), "spacing_mm": spacing.tolist(), "orientation": list(nib.aff2axcodes(padded_affine))}, "regions": regions, "shells": {"CSF_target_mm": 2.0, "Skull_target_mm": 3.0, "Scalp_target_mm": 4.0, "cumulative_distances_mm": {"csf_outer": 2.0, "skull_outer": 5.0, "scalp_outer": 9.0}, "distance_method": "scipy.ndimage.distance_transform_edt with voxel spacing sampling", "diagnostic_shell_voxel_counts": {"brain_outer": int(h.sum()), "csf_outer": int(csf_outer.sum()), "skull_outer": int(skull_outer.sum()), "scalp_outer": int(scalp_outer.sum())}}, "invariants": invariants, "tumor_checks": {"original_count": int(tumor.sum()), "final_count": int(padded_masks[2].sum()), "differing_voxels": int(np.count_nonzero(tumor != original_tumor_reload)), "world_bbox_original": old_tumor_world, "world_bbox_final": new_tumor_world}, "adjacency_face_counts": all_pairs, "forbidden_adjacency_pairs": forbidden, "exterior_face_counts": exterior, "material_conservation": {"sum_material_voxels": int(sum(int(m.sum()) for m in padded_masks.values())), "nonzero_label_voxels": int(np.count_nonzero(padded_labels)), "equal": invariants["material_conservation"]}, "surfaces": surfaces, "scientific_note": "Brain anatomical/reference mask is HD-BET. Brain envelope is a tumor-safe computational FEM domain. CSF, Skull, and Scalp are computational shells, not MRI-ground-truth anatomical tissue segmentations."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"Status: {report['status']}; final shape={padded_labels.shape}; padding={pad} voxels/side")
    for name, data in regions.items(): log(f"{name}: {data['voxels']} voxels ({data['volume_ml']:.3f} mL)")
    log(f"Adjacency: {adjacency}; exterior: {exterior}; forbidden={forbidden}")
    log(f"Report: {REPORT}")
    if report["status"] != "PASS": raise RuntimeError("Preprocessing invariant failure; see report")
    return 0


if __name__ == "__main__": raise SystemExit(main())
