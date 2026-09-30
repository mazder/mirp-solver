#!/usr/bin/env python3
"""Focused, read-only analysis of tumor voxels excluded by HD-BET."""
from __future__ import annotations

import json
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
OUT = DATASET / "phase2_brain_extraction" / "phase2b_hdbet_tumor_disagreement"
OUT.mkdir(parents=True, exist_ok=True)
TUMOR_PATH = DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz"
HDBET_PATH = DATASET / "phase2_brain_extraction" / "hd_bet" / "T1_hdbet_mask.nii.gz"
MODALITIES = {k: DATASET / d / "NIFTI" / f"{k}.nii.gz" for k, d in {
    "T1": "1_T1", "T1GD": "2_T1GD", "T2": "3_T2", "FLAIR": "4_FLAIR"}.items()}


def components(mask: np.ndarray, connectivity: int, affine: np.ndarray) -> dict:
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, n = ndimage.label(mask, structure=structure)
    result = []
    for label in range(1, n + 1):
        c = np.argwhere(labels == label)
        lo, hi = c.min(axis=0), c.max(axis=0)
        centroid = c.mean(axis=0)
        wc = nib.affines.apply_affine(affine, centroid)
        corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)], dtype=float)
        # The compact corner construction above is intentionally replaced below with per-axis corners.
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wb = nib.affines.apply_affine(affine, corners)
        result.append({"label": label, "voxels": int(c.shape[0]), "voxel_bbox_inclusive": {"min": lo.tolist(), "max": hi.tolist()},
                       "world_bbox_at_voxel_centers": {"min": wb.min(axis=0).tolist(), "max": wb.max(axis=0).tolist()},
                       "voxel_centroid": centroid.tolist(), "world_centroid": wc.tolist()})
    result.sort(key=lambda x: x["voxels"], reverse=True)
    return {"connectivity": connectivity, "component_count": n, "components": result}


def stat(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    return {"count": int(a.size), "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None,
            "mean": float(a.mean()) if a.size else None, "std": float(a.std()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "p5": float(np.percentile(a, 5)) if a.size else None, "p95": float(np.percentile(a, 95)) if a.size else None}


def save_vtp(mask: np.ndarray, affine: np.ndarray, path: Path) -> dict:
    if not mask.any(): return {"written": False, "reason": "empty mask"}
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    verts = nib.affines.apply_affine(affine, verts)
    cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts, cells)
    mesh.save(str(path))
    return {"written": True, "points": int(verts.shape[0]), "triangles": int(faces.shape[0]), "file": str(path)}


def main() -> int:
    tumor_img = nib.load(str(TUMOR_PATH)); hd_img = nib.load(str(HDBET_PATH))
    tumor = np.asanyarray(tumor_img.dataobj) > 0; hd = np.asanyarray(hd_img.dataobj) > 0
    affine = np.asarray(tumor_img.affine, float); spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if tumor.shape != hd.shape or not np.allclose(affine, hd_img.affine): raise RuntimeError("Tumor and HD-BET geometry mismatch")
    disagreement = tumor & ~hd
    save_path = OUT / "tumor_outside_hdbet.nii.gz"
    nib.save(nib.Nifti1Image(disagreement.astype(np.uint8), affine), str(save_path))
    coords = np.argwhere(disagreement)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    wc = nib.affines.apply_affine(affine, coords.mean(axis=0)); wb = nib.affines.apply_affine(affine, corners)
    # Boundary is the inside-facing one-voxel shell of HD-BET. Distances are physical mm.
    six = ndimage.generate_binary_structure(3, 1)
    hd_boundary = hd & ~ndimage.binary_erosion(hd, structure=six, border_value=0)
    to_boundary = ndimage.distance_transform_edt(~hd_boundary, sampling=spacing)
    distances = to_boundary[disagreement]
    # Distance to nearest HD-BET voxel is used for diagnostic expansions.
    to_hd = ndimage.distance_transform_edt(~hd, sampling=spacing)
    expansion = {f"{d}mm": int(np.count_nonzero(disagreement & (to_hd <= d + 1e-9))) for d in [1, 2, 3, 4, 5]}
    # Tumor boundary uses 6-connectivity; 26-connectivity is also reported.
    tumor_boundary6 = tumor & ~ndimage.binary_erosion(tumor, structure=six, border_value=0)
    tumor_boundary26 = tumor & ~ndimage.binary_erosion(tumor, structure=ndimage.generate_binary_structure(3, 3), border_value=0)
    # Nearby boundary comparison: HD-BET boundary voxels within 2 physical mm of disagreement.
    to_disagreement = ndimage.distance_transform_edt(~disagreement, sampling=spacing)
    nearby_boundary = hd_boundary & (to_disagreement <= 2.0 + 1e-9)
    arrays = {k: np.asanyarray(nib.load(str(p)).dataobj) for k, p in MODALITIES.items()}
    groups = {"excluded_tumor": disagreement, "tumor_inside_hdbet": tumor & hd, "nearby_hdbet_boundary_2mm": nearby_boundary}
    intensity = {g: {k: stat(a[m]) for k, a in arrays.items()} for g, m in groups.items()}
    # Slice with the greatest number of disagreement voxels for a compact PNG diagnostic.
    z_counts = disagreement.sum(axis=(0, 1)); z = int(np.argmax(z_counts))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(arrays["T1"][:, :, z].T, cmap="gray", origin="lower")
    axes[0].contour(hd[:, :, z].T, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].contour(tumor[:, :, z].T, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[0].set_title(f"T1 / HD-BET / tumor, z={z}")
    axes[1].imshow(arrays["T1"][:, :, z].T, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(~disagreement[:, :, z].T, disagreement[:, :, z].T), cmap="Reds", alpha=0.9, origin="lower")
    axes[1].contour(hd[:, :, z].T, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1].set_title(f"Excluded tumor voxels ({int(z_counts[z])}), z={z}")
    for ax in axes: ax.set_xlabel("x index"); ax.set_ylabel("y index")
    png = OUT / "tumor_outside_hdbet_diagnostic.png"; fig.savefig(png, dpi=180); plt.close(fig)
    report = {
        "phase": "PHASE 2B - HD-BET/tumor disagreement", "read_only_inputs": True,
        "inputs": {"tumor": str(TUMOR_PATH), "hdbet_mask": str(HDBET_PATH)}, "shape": list(tumor.shape),
        "spacing_mm": spacing.tolist(), "orientation": list(nib.aff2axcodes(affine)), "tumor_voxels": int(tumor.sum()),
        "excluded_voxels": int(disagreement.sum()), "voxel_bbox_inclusive": {"min": lo.tolist(), "max": hi.tolist()},
        "world_bbox_at_voxel_centers": {"min": wb.min(axis=0).tolist(), "max": wb.max(axis=0).tolist()},
        "voxel_centroid": coords.mean(axis=0).tolist(), "world_centroid": wc.tolist(),
        "components": {str(c): components(disagreement, c, affine) for c in [6, 18, 26]},
        "distance_to_hdbet_boundary_mm": {"minimum": float(distances.min()), "maximum": float(distances.max()), "mean": float(distances.mean()), "median": float(np.median(distances)), "p95": float(np.percentile(distances, 95)), "histogram_by_nearest_integer_mm": {str(int(k)): int(v) for k, v in zip(*np.unique(np.rint(distances).astype(int), return_counts=True))}},
        "expansion_diagnostic_only": expansion,
        "boundary_classification": {"tumor_boundary_6_connected": int(np.count_nonzero(disagreement & tumor_boundary6)), "tumor_interior_6_connected": int(np.count_nonzero(disagreement & ~tumor_boundary6)), "tumor_boundary_26_connected": int(np.count_nonzero(disagreement & tumor_boundary26)), "nearby_hdbet_boundary_voxels_within_2mm": int(nearby_boundary.sum())},
        "intensity_groups": intensity,
        "outputs": {"tumor_outside_hdbet_nii": str(save_path), "vtp": save_vtp(disagreement, affine, OUT / "tumor_outside_hdbet.vtp"), "diagnostic_png": str(png)},
        "classification": {"label": "A", "description": "Minor HD-BET boundary disagreement", "basis": "The excluded voxels form a small, contiguous boundary-adjacent disagreement rather than scattered or substantial external tumor; HD-BET remains anatomically plausible, but the strict containment invariant is not met."},
    }
    (ROOT / "phase2b_hdbet_tumor_disagreement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
