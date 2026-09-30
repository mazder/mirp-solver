#!/usr/bin/env python3
"""Build the tumor-safe computational brain envelope (Phase 2C only)."""
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
from skimage.measure import euler_number, marching_cubes


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
PHASE2_REPORT = ROOT / "phase2_brain_extraction_report.json"
TUMOR_PATH = DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz"
OUT = DATASET / "phase2_brain_extraction" / "phase2c_computational_brain_envelope"
REPORT = ROOT / "phase2c_computational_brain_envelope.json"


def log(message: str) -> None:
    print(f"[phase2c] {message}")


def discover_hdbet_path() -> Path:
    if PHASE2_REPORT.exists():
        report = json.loads(PHASE2_REPORT.read_text(encoding="utf-8"))
        source = report.get("candidates", {}).get("hd_bet", {}).get("source")
        if source:
            candidate = DATASET / "phase2_brain_extraction" / source
            if candidate.exists():
                return candidate
    candidates = sorted((DATASET / "phase2_brain_extraction" / "hd_bet").glob("*_mask.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError("Could not identify the recorded HD-BET mask from Phase 2 outputs")


def geometry_check(tumor_img: nib.Nifti1Image, h_img: nib.Nifti1Image) -> dict[str, object]:
    tumor_shape, h_shape = tuple(tumor_img.shape), tuple(h_img.shape)
    affine_match = bool(np.allclose(tumor_img.affine, h_img.affine, rtol=1e-5, atol=1e-5))
    tumor_spacing = np.linalg.norm(tumor_img.affine[:3, :3], axis=0)
    h_spacing = np.linalg.norm(h_img.affine[:3, :3], axis=0)
    spacing_match = bool(np.allclose(tumor_spacing, h_spacing, rtol=1e-5, atol=1e-5))
    orientation_match = tuple(nib.aff2axcodes(tumor_img.affine)) == tuple(nib.aff2axcodes(h_img.affine))
    result = {"both_3d": len(tumor_shape) == 3 and len(h_shape) == 3, "shape_match": tumor_shape == h_shape,
              "affine_match": affine_match, "spacing_match": spacing_match, "orientation_match": orientation_match,
              "tumor_shape": list(tumor_shape), "hdbet_shape": list(h_shape), "tumor_spacing_mm": tumor_spacing.tolist(),
              "hdbet_spacing_mm": h_spacing.tolist(), "tumor_orientation": list(nib.aff2axcodes(tumor_img.affine)),
              "hdbet_orientation": list(nib.aff2axcodes(h_img.affine))}
    if not all(result[k] for k in ["both_3d", "shape_match", "affine_match", "spacing_match", "orientation_match"]):
        raise RuntimeError(f"Geometry mismatch; no Boolean operation performed: {result}")
    return result


def world_bbox(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float]] | None:
    coords = np.argwhere(mask)
    if not coords.size:
        return None
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)
    world = nib.affines.apply_affine(affine, corners)
    return {"min": world.min(axis=0).tolist(), "max": world.max(axis=0).tolist()}


def bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    coords = np.argwhere(mask)
    return {"min": coords.min(axis=0).tolist(), "max": coords.max(axis=0).tolist()} if coords.size else None


def components(mask: np.ndarray, connectivity: int, affine: np.ndarray) -> list[dict[str, object]]:
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, count = ndimage.label(mask, structure=structure)
    result = []
    for label in range(1, count + 1):
        part = labels == label
        coords = np.argwhere(part)
        centroid = coords.mean(axis=0)
        result.append({"label": label, "voxels": int(part.sum()), "voxel_bbox_inclusive": bbox(part),
                       "world_bbox_at_voxel_centers": world_bbox(part, affine), "voxel_centroid": centroid.tolist(),
                       "world_centroid": nib.affines.apply_affine(affine, centroid).tolist()})
    return sorted(result, key=lambda x: int(x["voxels"]), reverse=True)


def save_surface(mask: np.ndarray, affine: np.ndarray, path: Path) -> dict[str, object]:
    if not mask.any():
        return {"written": False, "reason": "empty mask"}
    vertices, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    vertices = nib.affines.apply_affine(affine, vertices)
    cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    pv.PolyData(vertices, cells).save(str(path))
    return {"written": True, "file": str(path), "points": int(vertices.shape[0]), "triangles": int(faces.shape[0])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h_path = discover_hdbet_path()
    log(f"HD-BET anatomical/reference mask: {h_path}")
    log(f"Original tumor mask: {TUMOR_PATH}")
    tumor_img = nib.load(str(TUMOR_PATH))
    h_img = nib.load(str(h_path))
    geometry = geometry_check(tumor_img, h_img)
    tumor_first = np.asanyarray(tumor_img.dataobj) > 0
    # Independent reload is intentional: it checks that the authoritative source was not altered.
    tumor_second = np.asanyarray(nib.load(str(TUMOR_PATH)).dataobj) > 0
    h = np.asanyarray(h_img.dataobj) > 0
    if not np.array_equal(tumor_first, tumor_second):
        raise RuntimeError("Original tumor mask changed between independent read-only loads")
    added = tumor_first & ~h
    envelope = h | tumor_first
    spacing = np.asarray(geometry["tumor_spacing_mm"], dtype=float)
    six = ndimage.generate_binary_structure(3, 1)
    twenty_six = ndimage.generate_binary_structure(3, 3)
    tumor_diff = int(np.count_nonzero(tumor_first != tumor_second))
    outside_final = int(np.count_nonzero(tumor_first & ~envelope))
    h_outside_final = int(np.count_nonzero(h & ~envelope))
    minimal_difference = int(np.count_nonzero((envelope & ~h) != added))
    union_difference = int(np.count_nonzero(envelope != (h | tumor_first)))
    binary_ok = all(np.isin(x, [False, True]).all() for x in [added, envelope])
    # Adjacency and detached-component diagnostics.
    added_adj6 = added & ndimage.binary_dilation(h, structure=six)
    added_adj26 = added & ndimage.binary_dilation(h, structure=twenty_six)
    h_labels, h_count = ndimage.label(h, structure=six)
    env_labels, env_count = ndimage.label(envelope, structure=six)
    main_h_label = int(np.argmax(np.bincount(h_labels.ravel())[1:]) + 1) if h_count else 0
    detached = []
    for label in range(1, env_count + 1):
        part = env_labels == label
        if not np.any(part & h):
            detached.append(int(part.sum()))
    surface_h = h & ~ndimage.binary_erosion(h, structure=six, border_value=0)
    surface_env = envelope & ~ndimage.binary_erosion(envelope, structure=six, border_value=0)
    # Diagnostic visualization centered on the disagreement centroid.
    c = np.argwhere(added).mean(axis=0).astype(int)
    z = int(c[2])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # T1 is loaded only for visualization; it is not used to construct the envelope.
    t1 = np.asanyarray(nib.load(str(DATASET / "1_T1" / "NIFTI" / "T1.nii.gz")).dataobj)
    axes[0].imshow(t1[:, :, z].T, cmap="gray", origin="lower")
    axes[0].contour(h[:, :, z].T, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].contour(tumor_first[:, :, z].T, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[0].set_title(f"T1: HD-BET boundary + original tumor (z={z})")
    axes[1].imshow(t1[:, :, z].T, cmap="gray", origin="lower")
    axes[1].contour(envelope[:, :, z].T, levels=[0.5], colors="lime", linewidths=1.0)
    added_slice = np.argwhere(added[:, :, z])
    if added_slice.size:
        axes[1].scatter(added_slice[:, 0], added_slice[:, 1], c="red", s=16, marker="s", label="added tumor voxels")
        axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title(f"Tumor-safe envelope; added voxels={int(added[:, :, z].sum())}")
    for ax in axes: ax.set_xlabel("x index"); ax.set_ylabel("y index")
    png = OUT / "phase2c_envelope_diagnostic.png"; fig.savefig(png, dpi=180); plt.close(fig)
    envelope_path = OUT / "computational_brain_envelope.nii.gz"
    added_path = OUT / "added_tumor_to_hdbet.nii.gz"
    nib.save(nib.Nifti1Image(envelope.astype(np.uint8), tumor_img.affine), str(envelope_path))
    nib.save(nib.Nifti1Image(added.astype(np.uint8), tumor_img.affine), str(added_path))
    surface_paths = {"computational_brain_envelope": save_surface(envelope, tumor_img.affine, OUT / "computational_brain_envelope.vtp"),
                     "added_tumor_to_hdbet": save_surface(added, tumor_img.affine, OUT / "added_tumor_to_hdbet.vtp")}
    invariants = {"tumor_preservation": tumor_diff == 0, "complete_tumor_containment": outside_final == 0,
                  "hdbet_preservation": h_outside_final == 0, "minimal_addition": minimal_difference == 0,
                  "boolean_union_identity": union_difference == 0, "binary_integrity": binary_ok}
    report = {"phase": "PHASE 2C - tumor-safe computational brain envelope", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if all(invariants.values()) else "FAIL", "inputs": {"hdbet_mask": str(h_path), "original_tumor_mask": str(TUMOR_PATH)},
              "outputs": {"computational_brain_envelope": str(envelope_path), "added_tumor_to_hdbet": str(added_path), "diagnostic_png": str(png), "surfaces": surface_paths},
              "geometry": geometry, "voxel_spacing_mm": spacing.tolist(),
              "counts": {"hdbet_voxels": int(h.sum()), "original_tumor_voxels": int(tumor_first.sum()), "derived_reference_tumor_voxels": int(tumor_second.sum()), "added_tumor_voxels": int(added.sum()), "computational_envelope_voxels": int(envelope.sum())},
              "invariants": {**invariants, "tumor_differing_voxels": tumor_diff, "tumor_outside_final": outside_final, "hdbet_outside_final": h_outside_final, "minimal_addition_differing_voxels": minimal_difference, "union_identity_differing_voxels": union_difference},
              "bounding_boxes": {"hdbet_voxel": bbox(h), "hdbet_world": world_bbox(h, tumor_img.affine), "tumor_voxel": bbox(tumor_first), "tumor_world": world_bbox(tumor_first, tumor_img.affine), "added_voxel": bbox(added), "added_world": world_bbox(added, tumor_img.affine), "envelope_voxel": bbox(envelope), "envelope_world": world_bbox(envelope, tumor_img.affine)},
              "topology": {"hdbet_components_6": components(h, 6, tumor_img.affine), "envelope_components_6": components(envelope, 6, tumor_img.affine), "envelope_components_26": components(envelope, 26, tumor_img.affine), "hdbet_component_count_6": h_count, "envelope_component_count_6": env_count, "added_with_6_neighbor_in_hdbet": int(added_adj6.sum()), "added_without_6_neighbor_in_hdbet": int((added & ~added_adj6).sum()), "added_with_26_neighbor_in_hdbet": int(added_adj26.sum()), "added_without_26_neighbor_in_hdbet": int((added & ~added_adj26).sum()), "detached_envelope_component_sizes": detached, "surface_voxels_hdbet_6": int(surface_h.sum()), "surface_voxels_envelope_6": int(surface_env.sum()), "euler_hdbet_26": int(euler_number(h, connectivity=3)), "euler_envelope_26": int(euler_number(envelope, connectivity=3))},
              "scientific_interpretation": "The envelope is a derived tumor-safe computational domain (H union T), not a corrected HD-BET anatomical segmentation. No automatic dilation or topology repair was applied."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"Status: {report['status']}")
    log(f"HD-BET voxels: {h.sum()}; tumor voxels: {tumor_first.sum()}; added: {added.sum()}; envelope: {envelope.sum()}")
    log(f"Tumor outside final envelope: {outside_final}; detached components: {detached}")
    log(f"Report: {REPORT}")
    if report["status"] != "PASS": raise RuntimeError("Phase 2C invariant failure; see report")
    return 0


if __name__ == "__main__": raise SystemExit(main())
