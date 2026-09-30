#!/usr/bin/env python3
"""Read-only Phase 1 diagnostics for the MRI/tumor inputs."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "EGD-0001" / "MR_EGD-0001"
OUTPUT = ROOT / "phase1_data_diagnostics.json"

INPUTS = {
    "T1": DATASET / "1_T1" / "NIFTI" / "T1.nii.gz",
    "T1GD": DATASET / "2_T1GD" / "NIFTI" / "T1GD.nii.gz",
    "T2": DATASET / "3_T2" / "NIFTI" / "T2.nii.gz",
    "FLAIR": DATASET / "4_FLAIR" / "NIFTI" / "FLAIR.nii.gz",
    "MASK": DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz",
}


def world_point(affine: np.ndarray, index: np.ndarray) -> list[float]:
    point = affine @ np.r_[index.astype(float), 1.0]
    return point[:3].tolist()


def world_bbox(affine: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, list[float]]:
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=float,
    )
    transformed = nib.affines.apply_affine(affine, corners)
    return {"min": transformed.min(axis=0).tolist(), "max": transformed.max(axis=0).tolist()}


def finite_stats(data: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(data)
    values = np.asarray(data[finite], dtype=float)
    if values.size == 0:
        return {
            "finite_voxels": 0,
            "nonfinite_voxels": int(data.size),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {},
            "positive_voxels": 0,
            "nonzero_voxels": 0,
        }
    percentiles = np.percentile(values, [0, 1, 50, 99, 100])
    return {
        "finite_voxels": int(values.size),
        "nonfinite_voxels": int(data.size - values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "percentiles": {str(p): float(v) for p, v in zip([0, 1, 50, 99, 100], percentiles)},
        "positive_voxels": int(np.count_nonzero(values > 0)),
        "nonzero_voxels": int(np.count_nonzero(values != 0)),
    }


def image_info(name: str, path: Path) -> dict[str, object]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=float)
    matrix = affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    info: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "shape": list(data.shape),
        "ndim": int(data.ndim),
        "dtype": str(data.dtype),
        "header_dtype": str(image.header.get_data_dtype()),
        "affine": affine.tolist(),
        "orientation": list(nib.aff2axcodes(affine)),
        "spacing_mm": spacing.tolist(),
        "voxel_volume_mm3": float(abs(np.linalg.det(matrix))),
        "stats": finite_stats(data),
    }
    if name == "MASK":
        finite = np.isfinite(data)
        tumor = finite & (data > 0)
        coords = np.argwhere(tumor)
        info["tumor_definition"] = "finite voxel values > 0 (no modification applied)"
        info["tumor_voxel_count"] = int(coords.shape[0])
        info["tumor_volume_mm3"] = float(coords.shape[0] * abs(np.linalg.det(matrix)))
        info["tumor_volume_ml"] = float(coords.shape[0] * abs(np.linalg.det(matrix)) / 1000.0)
        if coords.size:
            lo = coords.min(axis=0)
            hi = coords.max(axis=0)
            centroid_voxel = coords.mean(axis=0)
            info["tumor_voxel_bbox_inclusive"] = {"min": lo.tolist(), "max": hi.tolist()}
            info["tumor_world_bbox_at_voxel_centers"] = world_bbox(affine, lo, hi)
            info["tumor_centroid_voxel"] = centroid_voxel.tolist()
            info["tumor_centroid_world"] = world_point(affine, centroid_voxel)
        else:
            info["tumor_voxel_bbox_inclusive"] = None
            info["tumor_world_bbox_at_voxel_centers"] = None
            info["tumor_centroid_voxel"] = None
            info["tumor_centroid_world"] = None
        unique, counts = np.unique(data[finite], return_counts=True)
        if unique.size <= 32:
            info["finite_value_histogram"] = {
                str(float(value)): int(count) for value, count in zip(unique, counts)
            }
        else:
            info["finite_unique_value_count"] = int(unique.size)
    return info


def main() -> int:
    images = {name: image_info(name, path) for name, path in INPUTS.items()}
    reference = images["T1"]
    mismatches: list[dict[str, object]] = []
    for name, info in images.items():
        shape_ok = info["shape"] == reference["shape"]
        affine_ok = np.allclose(info["affine"], reference["affine"], rtol=1e-5, atol=1e-5)
        ndim_ok = info["ndim"] == 3
        if not (shape_ok and affine_ok and ndim_ok):
            mismatches.append(
                {"name": name, "shape_match": shape_ok, "affine_match": affine_ok, "three_dimensional": ndim_ok}
            )

    report = {
        "phase": "PHASE 1 - data inspection",
        "read_only_inputs": True,
        "dataset": str(DATASET),
        "inputs": images,
        "geometry_validation": {
            "reference": "T1",
            "all_three_dimensional": not mismatches,
            "all_shapes_match": all(info["shape"] == reference["shape"] for info in images.values()),
            "all_affines_match": all(
                np.allclose(info["affine"], reference["affine"], rtol=1e-5, atol=1e-5)
                for info in images.values()
            ),
            "aligned": not mismatches,
            "mismatches": mismatches,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("PHASE 1 MRI DATA DIAGNOSTICS (read-only)")
    print(f"Dataset: {DATASET}")
    for name, info in images.items():
        stats = info["stats"]
        print(
            f"{name:5s}: shape={tuple(info['shape'])}, spacing={tuple(round(x, 6) for x in info['spacing_mm'])} mm, "
            f"orientation={''.join(info['orientation'])}, dtype={info['dtype']}, "
            f"range=[{stats['min']}, {stats['max']}], finite={stats['finite_voxels']}"
        )
    mask = images["MASK"]
    print(f"Geometry aligned: {report['geometry_validation']['aligned']}")
    print(f"Tumor voxels (finite MASK > 0): {mask['tumor_voxel_count']}")
    print(f"Tumor volume: {mask['tumor_volume_mm3']} mm^3 ({mask['tumor_volume_ml']} mL)")
    print(f"Tumor voxel bbox (inclusive): {mask['tumor_voxel_bbox_inclusive']}")
    print(f"Tumor world bbox: {mask['tumor_world_bbox_at_voxel_centers']}")
    print(f"Tumor centroid voxel: {mask['tumor_centroid_voxel']}")
    print(f"Tumor centroid world: {mask['tumor_centroid_world']}")
    print(f"Report: {OUTPUT}")
    return 0 if report["geometry_validation"]["aligned"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
