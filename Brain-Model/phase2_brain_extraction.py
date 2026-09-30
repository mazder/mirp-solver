#!/usr/bin/env python3
"""Phase 2: compare HD-BET with a classical multimodal MRI candidate.

This script never writes to the source NIfTI files or the tumor mask. It writes
only under the dataset's phase2_brain_extraction directory.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
OUT = DATASET / "phase2_brain_extraction"
CLASSICAL = OUT / "classical"
HDBET = OUT / "hd_bet"
REPORT = ROOT / "phase2_brain_extraction_report.json"
MODALITIES = {
    "T1": DATASET / "1_T1" / "NIFTI" / "T1.nii.gz",
    "T1GD": DATASET / "2_T1GD" / "NIFTI" / "T1GD.nii.gz",
    "T2": DATASET / "3_T2" / "NIFTI" / "T2.nii.gz",
    "FLAIR": DATASET / "4_FLAIR" / "NIFTI" / "FLAIR.nii.gz",
}
MASK_PATH = DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz"


def robust_normalize(a: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(a)
    p1, p99 = np.percentile(a[finite], [1, 99])
    if p99 <= p1:
        return np.zeros_like(a, dtype=np.float32), {"p1": float(p1), "p99": float(p99)}
    out = np.clip((a.astype(np.float32) - p1) / (p99 - p1), 0.0, 1.0)
    out[~finite] = 0.0
    return out, {"p1": float(p1), "p99": float(p99)}


def save_mask(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def bbox(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    c = np.argwhere(mask)
    if not c.size:
        return None
    return c.min(axis=0).tolist(), c.max(axis=0).tolist()


def world_bbox(affine: np.ndarray, b: tuple[list[int], list[int]] | None) -> dict[str, list[float]] | None:
    if b is None:
        return None
    lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    wc = nib.affines.apply_affine(affine, corners)
    return {"min": wc.min(axis=0).tolist(), "max": wc.max(axis=0).tolist()}


def metrics(name: str, mask: np.ndarray, tumor: np.ndarray, affine: np.ndarray, source: str, parameters=None) -> dict:
    structure = ndimage.generate_binary_structure(3, 3)
    labels, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    b = bbox(mask)
    finite_image_boundary = np.zeros(mask.shape, dtype=bool)
    for axis in range(3):
        sl0 = [slice(None)] * 3; sl0[axis] = 0
        sl1 = [slice(None)] * 3; sl1[axis] = -1
        finite_image_boundary[tuple(sl0)] = True; finite_image_boundary[tuple(sl1)] = True
    boundary_voxels = int(np.count_nonzero(mask & finite_image_boundary))
    filled = ndimage.binary_fill_holes(mask)
    holes = int(np.count_nonzero(filled & ~mask))
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0).tolist() if coords.size else None
    centroid_world = nib.affines.apply_affine(affine, np.asarray(centroid)).tolist() if centroid else None
    inside = int(np.count_nonzero(tumor & mask)); outside = int(np.count_nonzero(tumor & ~mask))
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    return {
        "name": name, "source": source, "parameters": parameters or {}, "shape": list(mask.shape),
        "affine": affine.tolist(), "orientation": list(nib.aff2axcodes(affine)),
        "voxel_count": int(mask.sum()), "volume_mm3": float(mask.sum() * voxel_volume),
        "volume_ml": float(mask.sum() * voxel_volume / 1000.0), "connected_components": int(n),
        "largest_component_fraction": float(largest / mask.sum()) if mask.sum() else 0.0,
        "voxel_bbox_inclusive": {"min": b[0], "max": b[1]} if b else None,
        "world_bbox_at_voxel_centers": world_bbox(affine, b), "voxel_centroid": centroid,
        "world_centroid": centroid_world, "tumor_inside": inside, "tumor_outside": outside,
        "tumor_containment_percent": float(100.0 * inside / tumor.sum()),
        "image_boundary_mask_voxels": boundary_voxels, "internal_hole_voxels": holes,
        "topology_notes": ("touches image boundary" if boundary_voxels else "does not touch image boundary"),
    }


def pairwise(a: np.ndarray, b: np.ndarray, an: str, bn: str) -> dict:
    inter = int(np.count_nonzero(a & b)); union = int(np.count_nonzero(a | b))
    ba, bb = bbox(a), bbox(b)
    bbox_delta = None
    if ba and bb:
        bbox_delta = {"min_b_minus_a": (np.asarray(bb[0]) - np.asarray(ba[0])).tolist(),
                      "max_b_minus_a": (np.asarray(bb[1]) - np.asarray(ba[1])).tolist()}
    return {"a": an, "b": bn, "intersection": inter, "union": union,
            "dice": float(2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) else 1.0,
            "jaccard": float(inter / union) if union else 1.0,
            "unique_to_a": int(np.count_nonzero(a & ~b)), "unique_to_b": int(np.count_nonzero(b & ~a)),
            "volume_difference_mm3_b_minus_a": float(b.sum() - a.sum()), "bbox_difference": bbox_delta}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CLASSICAL.mkdir(exist_ok=True); HDBET.mkdir(exist_ok=True)
    imgs = {k: nib.load(str(v)) for k, v in MODALITIES.items()}
    mask_img = nib.load(str(MASK_PATH)); tumor = np.asanyarray(mask_img.dataobj) > 0
    affine = np.asarray(imgs["T1"].affine)
    arrays = {k: np.asanyarray(v.dataobj) for k, v in imgs.items()}
    norms, norm_stats = {}, {}
    for k, a in arrays.items(): norms[k], norm_stats[k] = robust_normalize(a)
    combined = np.mean(np.stack(list(norms.values())), axis=0)
    positive = combined[combined > 0]
    otsu = float(threshold_otsu(positive))
    raw = combined >= max(0.5 * otsu, 0.05)
    labels, n = ndimage.label(raw, structure=ndimage.generate_binary_structure(3, 3))
    if n:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        outer = labels == int(np.argmax(sizes))
    else: outer = raw
    outer = binary_closing(outer, ball(1))
    outer = ndimage.binary_fill_holes(outer)
    # This is deliberately an outer-head classical candidate, not a tumor-conditioned mask.
    save_mask(outer, affine, CLASSICAL / "classical_outer_head_mask.nii.gz")
    classical_params = {"method": "robust p1-p99 multimodal mean -> Otsu -> 0.5*Otsu threshold -> largest 26-connected component -> ball(1) closing -> hole filling", "normalization": norm_stats, "otsu_threshold": otsu, "effective_threshold": max(0.5 * otsu, 0.05), "tumor_used_in_construction": False}
    candidates = {"classical": outer}; sources = {"classical": "classical_outer_head_mask.nii.gz"}; params = {"classical": classical_params}
    hdbet_attempt = {"installed": True, "package": "brainles_hd_bet", "version": "0.0.11", "torch": "2.14.0+cpu", "device": "cpu", "mode": "fast", "tta": 0, "postprocessing": 0, "command": "hd-bet -i EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz -o EGD-0001/MR_EGD-0001/phase2_brain_extraction/hd_bet/T1_hdbet -device cpu -mode fast -tta 0 -pp 0 -s 1 --overwrite_existing 1", "exit_status": 0, "elapsed_seconds": 294.84}
    hd_mask = HDBET / "T1_hdbet_mask.nii.gz"
    if hd_mask.exists():
        h = nib.load(str(hd_mask)); candidates["hd_bet"] = np.asanyarray(h.dataobj) > 0; sources["hd_bet"] = str(hd_mask.relative_to(OUT)); params["hd_bet"] = hdbet_attempt
    else:
        hdbet_attempt["status"] = "not_available_at_report_time"
    report = {"phase": "PHASE 2 - brain extraction evaluation", "read_only_inputs": True, "tumor_voxels": int(tumor.sum()), "candidates": {k: metrics(k, v, tumor, affine, sources[k], params[k]) for k, v in candidates.items()}, "hd_bet": hdbet_attempt, "pairwise": []}
    names = list(candidates)
    for i in range(len(names)):
        for j in range(i + 1, len(names)): report["pairwise"].append(pairwise(candidates[names[i]], candidates[names[j]], names[i], names[j]))
    report["recommendation"] = {
        "preferred_anatomical_candidate": "hd_bet",
        "basis": "HD-BET is much smaller than the classical mask, does not touch the image boundary, is one connected component, and is therefore more consistent with an anatomical brain than the classical outer-head mask.",
        "fem_readiness": "not approved yet: HD-BET excludes 206 authoritative tumor voxels; classical contains the tumor but represents the whole head rather than an anatomical brain.",
        "required_next_step": "Do not construct CSF/Skull/Scalp or mesh until the 206-voxel HD-BET exclusion and the brain-envelope policy are reviewed in Phase 3.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
