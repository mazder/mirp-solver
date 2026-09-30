#!/usr/bin/env python3
"""Phase 3B: tumor-safe FEM support and monotonic computational shells."""
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
HDBET = DATASET / "phase2_brain_extraction" / "hd_bet" / "T1_hdbet_mask.nii.gz"
TUMOR = DATASET / "5_MASK" / "NIFTI" / "MASK.nii.gz"
OUT = DATASET / "preprocessing" / "phase3b_results"
REPORT = OUT / "phase3b_preprocess_report.json"


def log(s: str) -> None:
    print(f"[phase3b] {s}")


def grow_physical(mask: np.ndarray, distance_mm: float, spacing: np.ndarray) -> np.ndarray:
    distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
    return mask | (distance <= distance_mm + 1e-12)


def synthetic_grow_test() -> dict[str, object]:
    mask = np.zeros((9, 9, 9), dtype=bool); mask[4, 4, 4] = True
    spacing = np.ones(3, dtype=float)
    grown = grow_physical(mask, 1.0, spacing)
    six = ndimage.generate_binary_structure(3, 1)
    face_neighbors = ndimage.binary_dilation(mask, structure=six) & ~mask
    passed = bool(np.all(~mask | grown) and np.all(face_neighbors <= grown) and int(grown.sum()) == 7)
    if not passed: raise RuntimeError("Synthetic grow_physical test failed")
    return {"passed": passed, "original_voxels": 1, "grown_voxels": int(grown.sum()), "face_neighbors_captured": int(np.count_nonzero(face_neighbors & grown))}


def bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    c = np.argwhere(mask)
    return {"min": c.min(0).tolist(), "max": c.max(0).tolist()} if c.size else None


def world_bbox(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float]] | None:
    c = np.argwhere(mask)
    if not c.size: return None
    lo, hi = c.min(0), c.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    w = nib.affines.apply_affine(affine, corners)
    return {"min": w.min(0).tolist(), "max": w.max(0).tolist()}


def centroid(mask: np.ndarray, affine: np.ndarray) -> dict[str, list[float] | None]:
    c = np.argwhere(mask)
    if not c.size: return {"voxel": None, "world": None}
    v = c.mean(0)
    return {"voxel": v.tolist(), "world": nib.affines.apply_affine(affine, v).tolist()}


def geometry(mask: np.ndarray, affine: np.ndarray) -> dict[str, object]:
    n = int(mask.sum()); vv = abs(float(np.linalg.det(affine[:3, :3])))
    return {"voxels": n, "volume_mm3": n * vv, "volume_ml": n * vv / 1000.0, "voxel_bbox": bbox(mask), "world_bbox": world_bbox(mask, affine), "centroid": centroid(mask, affine)}


def face_adjacency(labels: np.ndarray) -> dict[str, int]:
    result = {f"{i}-{j}": 0 for i in range(6) for j in range(i + 1, 6)}
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        d = a != b
        for x, y in zip(a[d].ravel(), b[d].ravel()):
            result[f"{min(int(x), int(y))}-{max(int(x), int(y))}"] += 1
    return result


def exterior_face_counts(labels: np.ndarray) -> tuple[dict[str, int], int, int]:
    bg = labels == 0; six = ndimage.generate_binary_structure(3, 1)
    bg_labels, n = ndimage.label(bg, structure=six)
    boundary_ids = set(np.concatenate([bg_labels[0].ravel(), bg_labels[-1].ravel(), bg_labels[:, 0].ravel(), bg_labels[:, -1].ravel(), bg_labels[:, :, 0].ravel(), bg_labels[:, :, -1].ravel()]).tolist()) - {0}
    exterior = np.isin(bg_labels, list(boundary_ids))
    counts = {str(i): 0 for i in range(6)}
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis); b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        ea = np.take(exterior, range(exterior.shape[axis] - 1), axis=axis); eb = np.take(exterior, range(1, exterior.shape[axis]), axis=axis)
        for material in range(1, 6): counts[str(material)] += int(np.count_nonzero(((a == material) & (b == 0) & eb) | ((b == material) & (a == 0) & ea)))
    internal = bg & ~exterior; internal_labels, internal_count = ndimage.label(internal, structure=six)
    return counts, int(internal_count), int(internal.sum())


def surface(mask: np.ndarray, affine: np.ndarray, path: Path) -> dict[str, object]:
    v, f, _, _ = marching_cubes(mask.astype(np.float32), 0.5); v = nib.affines.apply_affine(affine, v)
    cells = np.hstack([np.full((len(f), 1), 3, np.int64), f.astype(np.int64)]).ravel(); pv.PolyData(v, cells).save(str(path))
    return {"file": str(path), "points": int(len(v)), "triangles": int(len(f))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_grow_test(); log("Synthetic grow_physical test: PASS")
    h_img, t_img = nib.load(str(HDBET)), nib.load(str(TUMOR))
    h = np.asanyarray(h_img.dataobj) > 0; tumor = np.asanyarray(t_img.dataobj) > 0
    if h.ndim != 3 or tumor.ndim != 3 or h.shape != tumor.shape or not np.allclose(h_img.affine, t_img.affine): raise RuntimeError("HD-BET/tumor geometry mismatch")
    affine = np.asarray(h_img.affine, float); spacing = np.linalg.norm(affine[:3, :3], axis=0); six = ndimage.generate_binary_structure(3, 1)
    tumor_reload = np.asanyarray(nib.load(str(TUMOR)).dataobj) > 0
    if not np.array_equal(tumor, tumor_reload): raise RuntimeError("Tumor changed between read-only loads")
    original_tumor_world = world_bbox(tumor, affine); original_tumor_centroid = centroid(tumor, affine)
    face_neighbors = ndimage.binary_dilation(tumor, structure=six) & ~tumor
    support = face_neighbors & ~h
    envelope = h | tumor | support
    tumor_exposed_before = int(face_count(tumor, (~h) & ~tumor))
    tumor_support_faces = int(face_count(tumor, support))
    tumor_exposed_after = int(face_count(tumor, (~envelope) & ~tumor))
    pad = 11; pw = [(pad, pad)] * 3
    hp, tp, ep = np.pad(h, pw), np.pad(tumor, pw), np.pad(envelope, pw)
    ap = affine.copy(); ap[:3, 3] -= affine[:3, :3] @ np.full(3, pad)
    csf_outer = grow_physical(ep, 2.0, spacing); skull_outer = grow_physical(csf_outer, 3.0, spacing); scalp_outer = grow_physical(skull_outer, 4.0, spacing)
    masks = {1: ep & ~tp, 2: tp, 3: csf_outer & ~ep, 4: skull_outer & ~csf_outer, 5: scalp_outer & ~skull_outer}
    labels = np.zeros(ep.shape, np.uint8)
    for i, m in masks.items(): labels[m] = i
    adjacency = face_adjacency(labels); exterior, internal_components, internal_voxels = exterior_face_counts(labels)
    allowed = {"1-2", "1-3", "3-4", "4-5"}; forbidden = {k: v for k, v in adjacency.items() if k not in allowed and not k.startswith("0-") and v}
    invariants = {"tumor_count": int(tumor.sum()) == 82652, "tumor_differing_voxels_zero": int(np.count_nonzero(tumor != tumor_reload)) == 0, "tumor_exact_equal": bool(np.array_equal(tumor, tumor_reload),), "tumor_inside_envelope": int(np.count_nonzero(tumor & ~envelope)) == 0, "support_minimal_definition": int(np.count_nonzero((envelope & ~h) != (tumor | support) & ~h)) == 0, "cumulative_nesting": bool(np.all(ep <= csf_outer) and np.all(csf_outer <= skull_outer) and np.all(skull_outer <= scalp_outer)), "mutually_exclusive": int(np.count_nonzero(sum(masks.values()) > 1)) == 0, "no_forbidden_face_adjacency": not forbidden, "brain_exterior_zero": exterior["1"] == 0, "tumor_exterior_zero": exterior["2"] == 0, "csf_exterior_zero": exterior["3"] == 0, "skull_exterior_zero": exterior["4"] == 0, "scalp_exterior_positive": exterior["5"] > 0, "material_conservation": int(sum(m.sum() for m in masks.values())) == int(np.count_nonzero(labels)), "world_tumor_bbox_preserved": world_bbox(np.pad(tumor, pw), ap) == original_tumor_world, "world_tumor_centroid_preserved": np.allclose(centroid(np.pad(tumor, pw), ap)["world"], original_tumor_centroid["world"])}
    outputs = {}
    for name, arr in {"brain_anatomical_hdbet": hp, "brain_envelope_fem": ep, "brain_mask": ep, "tumor_mask": tp, "csf_mask": masks[3], "skull_mask": masks[4], "scalp_mask": masks[5]}.items():
        p = OUT / f"{name}.nii.gz"; nib.save(nib.Nifti1Image(arr.astype(np.uint8), ap), str(p)); outputs[name] = str(p)
    p = OUT / "material_labels.nii.gz"; nib.save(nib.Nifti1Image(labels, ap), str(p)); outputs["material_labels"] = str(p)
    # Labels on the unpadded geometry are retained for coordinate/audit comparison.
    unpadded = np.zeros(envelope.shape, np.uint8); unpadded[ep[pad:-pad, pad:-pad, pad:-pad] & ~tp[pad:-pad, pad:-pad, pad:-pad]] = 1; unpadded[ tumor ] = 2
    unpadded[csf_outer[pad:-pad, pad:-pad, pad:-pad] & ~envelope] = 3; unpadded[skull_outer[pad:-pad, pad:-pad, pad:-pad] & ~csf_outer[pad:-pad, pad:-pad, pad:-pad]] = 4; unpadded[scalp_outer[pad:-pad, pad:-pad, pad:-pad] & ~skull_outer[pad:-pad, pad:-pad, pad:-pad]] = 5
    p = OUT / "material_labels_original_geometry.nii.gz"; nib.save(nib.Nifti1Image(unpadded, affine), str(p)); outputs["material_labels_original_geometry"] = str(p)
    # Diagnostic three-plane overlay at the Phase 2B disagreement centroid.
    disagreement = tumor & ~h; c = np.argwhere(disagreement).mean(0).astype(int); t1 = np.asanyarray(nib.load(str(DATASET / "1_T1/NIFTI/T1.nii.gz")).dataobj)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (name, axis, idx) in zip(axes, [("axial", 2, int(c[2])), ("coronal", 1, int(c[1])), ("sagittal", 0, int(c[0]))]):
        sl = [slice(None)] * 3; sl[axis] = idx; ax.imshow(t1[tuple(sl)].T, cmap="gray", origin="lower")
        for i, color in {1:"cyan",2:"yellow",3:"blue",4:"orange",5:"red"}.items(): ax.contour(masks[i][tuple(sl)].T, levels=[0.5], colors=color, linewidths=0.8)
        ax.set_title(f"{name} index={idx}")
    png = OUT / "phase3b_material_diagnostic.png"; fig.savefig(png, dpi=180); plt.close(fig); outputs["diagnostic_png"] = str(png)
    surfaces = {n: surface(a, ap, OUT / f"{n}.vtp") for n, a in {"brain_envelope_fem": ep, "csf_outer": csf_outer, "skull_outer": skull_outer, "scalp_outer": scalp_outer}.items()}
    regions = {n: geometry(m, ap) for n, m in {"BrainTissue": masks[1], "Tumor": masks[2], "CSF": masks[3], "Skull": masks[4], "Scalp": masks[5]}.items()}
    failure = [] if all(invariants.values()) else [k for k, v in invariants.items() if not v]
    inter = int(np.count_nonzero(h & envelope)); union = int(np.count_nonzero(h | envelope))
    similarity = {"intersection": inter, "union": union, "dice": float(2 * inter / (int(h.sum()) + int(envelope.sum()))), "jaccard": float(inter / union)}
    report = {"phase": "PHASE 3B derived material preprocessing (no meshing)", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if not failure else "FAIL", "failure_invariants": failure, "inputs": {"hdbet": str(HDBET), "original_tumor": str(TUMOR)}, "outputs": outputs, "original_affine": affine.tolist(), "final_affine": ap.tolist(), "original_shape": list(h.shape), "final_shape": list(labels.shape), "spacing_mm": spacing.tolist(), "padding_voxels_each_side": pad, "synthetic_grow_test": synthetic, "counts": {"hdbet": int(h.sum()), "tumor": int(tumor.sum()), "hdbet_union_tumor": int((h | tumor).sum()), "support_added": int(support.sum()), "fem_envelope": int(envelope.sum()), "final_brain_envelope": int(ep.sum())}, "hdbet_vs_final_envelope": similarity, "support": {"voxels": int(support.sum()), "volume_mm3": float(support.sum() * abs(np.linalg.det(affine[:3,:3]))), "voxel_bbox": bbox(support), "world_bbox": world_bbox(support, affine), "distance_to_hdbet_mm": {"min": float(ndimage.distance_transform_edt(~h, sampling=spacing)[support].min()), "max": float(ndimage.distance_transform_edt(~h, sampling=spacing)[support].max())}, "tumor_faces_exposed_before": tumor_exposed_before, "tumor_support_faces": tumor_support_faces, "tumor_faces_exposed_after": tumor_exposed_after}, "regions": regions, "cumulative_counts": {"brain_outer": int(ep.sum()), "csf_outer": int(csf_outer.sum()), "skull_outer": int(skull_outer.sum()), "scalp_outer": int(scalp_outer.sum())}, "tumor_preservation": {"original_count": int(tumor.sum()), "final_count": int(tp.sum()), "differing_voxels": int(np.count_nonzero(tumor != tumor_reload)), "exact_equality": bool(np.array_equal(tumor, tumor_reload)), "original_world_bbox": original_tumor_world, "final_world_bbox": world_bbox(tp, ap), "original_world_centroid": original_tumor_centroid["world"], "final_world_centroid": centroid(tp, ap)["world"]}, "adjacency_face_counts": adjacency, "forbidden_adjacency_counts": forbidden, "exterior_face_counts": exterior, "internal_background_components": internal_components, "internal_background_voxels": internal_voxels, "material_conservation": {"sum_material_voxels": int(sum(m.sum() for m in masks.values())), "nonzero_labels": int(np.count_nonzero(labels)), "equal": invariants["material_conservation"]}, "invariants": invariants, "surfaces": surfaces, "scientific_note": "HD-BET remains the anatomical/reference mask. The FEM envelope includes exact tumor voxels and a minimal computational BrainTissue support layer. CSF, Skull, and Scalp are computational shells, not MRI-ground-truth anatomical segmentations."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Status: {report['status']}; support={support.sum()}; envelope={ep.sum()}; regions={[(k, v['voxels']) for k,v in regions.items()]}")
    log(f"Adjacency={adjacency}; exterior={exterior}; internal_background={internal_components}/{internal_voxels}")
    log(f"Report: {REPORT}")
    if failure: raise RuntimeError(f"Phase 3B invariant failure: {failure}")
    return 0


def face_count(a: np.ndarray, b: np.ndarray) -> int:
    total = 0
    for axis in range(3):
        x = np.take(a, range(a.shape[axis] - 1), axis=axis); y = np.take(b, range(1, b.shape[axis]), axis=axis)
        total += int(np.count_nonzero(x & y))
        x = np.take(b, range(a.shape[axis] - 1), axis=axis); y = np.take(a, range(1, a.shape[axis]), axis=axis)
        total += int(np.count_nonzero(x & y))
    return total


if __name__ == "__main__": raise SystemExit(main())
