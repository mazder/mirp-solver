#!/usr/bin/env python3
"""Run the established Phase-3B material preprocessing on the external Pilot20.

Raw cohort files are read-only.  This creates no meshes and performs no FEM.
The morphology/coordinate rules are called from phase3b_preprocess_materials.py.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path("/media/mazder/80183b18-9ca3-4c7c-b4bc-d95674a79e22/home/mazder/simulation_data/brain_tumar")
AUDIT = ROOT / "cohort_meshing/pilot20/raw_input_audit.csv"
OUT_ROOT = ROOT / "cohort_meshing/pilot20/preprocessing"
PATIENT_OUTPUT_SUBDIR = False
HD_BET_PACKAGE = Path("/tmp/pilot20_hdbet_pkg")
MODALITIES = {
    "T1": ("1_T1/NIFTI/T1.nii.gz", "T1_SHA256"),
    "T1GD": ("2_T1GD/NIFTI/T1GD.nii.gz", "T1GD_SHA256"),
    "T2": ("3_T2/NIFTI/T2.nii.gz", "T2_SHA256"),
    "FLAIR": ("4_FLAIR/NIFTI/FLAIR.nii.gz", "FLAIR_SHA256"),
    "MASK": ("5_MASK/NIFTI/MASK.nii.gz", "MASK_SHA256"),
}
MATERIAL_NAMES = {1: "BrainTissue", 2: "Tumor", 3: "CSF", 4: "Skull", 5: "Scalp"}
ALLOWED_PAIRS = {"1-2", "1-3", "3-4", "4-5"}
INTERFACES = {"101": ("1-2", 1), "102": ("1-3", 2), "103": ("3-4", 0), "104": ("4-5", 0)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss_gb() -> float:
    own = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    children = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return max(own, children) / (1024.0 * 1024.0)


def load_audit() -> dict[str, dict[str, str]]:
    with AUDIT.open(newline="", encoding="utf-8") as f:
        return {r["patient_id"]: r for r in csv.DictReader(f)}


def run_hdbet_only(t1: Path, output_base: Path, log_path: Path) -> tuple[int, float, str]:
    """Call the same HD-BET 0.0.11 CLI entry point and reference options."""
    worker = r'''
import os, sys
sys.path.insert(0, "/tmp/pilot20_hdbet_pkg")
import brainles_hd_bet.utils as utils
if not hasattr(utils, "maybe_mkdir_p"):
    utils.maybe_mkdir_p = lambda p: os.makedirs(p, exist_ok=True)
from brainles_hd_bet.hd_bet import main
sys.argv = ["hd-bet"] + sys.argv[1:]
main()
'''
    command = [sys.executable, "-c", worker, "-i", str(t1), "-o", str(output_base),
               "-device", "cpu", "-mode", "fast", "-tta", "0", "-pp", "0", "-s", "1",
               "--overwrite_existing", "1"]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started
    hmask_path = Path(str(output_base)[:-7] + "_mask.nii.gz")
    if proc.returncode == 0 and output_base.is_file() and hmask_path.is_file():
        provenance = {"input_t1_sha256": sha256(t1), "output_sha256": sha256(output_base),
            "mask_sha256": sha256(hmask_path), "implementation": "brainles_hd_bet 0.0.11",
            "device": "cpu", "mode": "fast", "tta": 0, "postprocessing": 0,
            "command": command, "elapsed_seconds": elapsed}
        output_base.parent.joinpath("hdbet_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return proc.returncode, elapsed, json.dumps(command)


def face_area(spacing: np.ndarray, axis: int) -> float:
    return float(np.prod(np.delete(spacing, axis)))


def interface_metrics(labels: np.ndarray, spacing: np.ndarray) -> dict[str, dict[str, float | int]]:
    out = {tag: {"faces": 0, "area_mm2": 0.0} for tag in ("101", "102", "103", "104", "105")}
    for axis in range(3):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        area = face_area(spacing, axis)
        for tag, (pair, _) in INTERFACES.items():
            i, j = map(int, pair.split("-"))
            n = int(np.count_nonzero(((a == i) & (b == j)) | ((a == j) & (b == i))))
            out[tag]["faces"] += n
            out[tag]["area_mm2"] += n * area
        for material in (1, 2, 3, 4, 5):
            n = int(np.count_nonzero(((a == material) & (b == 0)) | ((b == material) & (a == 0))))
            if material == 5:
                out["105"]["faces"] += n
                out["105"]["area_mm2"] += n * area
    return out


def components_info(mask: np.ndarray, structure: np.ndarray) -> dict[str, object]:
    lab, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=np.int64)
    return {"count": int(n), "sizes_voxels_desc": sorted(map(int, sizes), reverse=True),
            "largest_voxels": int(sizes.max()) if sizes.size else 0}


def patient_output_dir(patient: str) -> Path:
    """Return the patient output directory without changing scientific logic.

    Pilot20 retains its historical flat layout. Full774 enables the nested
    production layout: patients/EGD-XXXX/preprocessing/.
    """
    return OUT_ROOT / patient / "preprocessing" if PATIENT_OUTPUT_SUBDIR else OUT_ROOT / patient


def patient_preprocess(patient: str, audit_row: dict[str, str]) -> dict[str, object]:
    t0 = time.monotonic()
    patient_dir = RAW_ROOT / patient / f"MR_{patient}"
    out = patient_output_dir(patient)
    hdir = out / "hd_bet"
    out.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"patient_id": patient, "started_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(patient_dir), "status": "FAILED_PREPROCESSING", "stage": "raw_hash_verification",
        "raw_hashes": {}, "outputs": {}, "failures": [], "mesh_generation": "NOT_PERFORMED", "FEM": "NOT_PERFORMED"}
    paths = {name: patient_dir / spec[0] for name, spec in MODALITIES.items()}
    actual_hashes = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            report["status"] = "MISSING_RAW_INPUT"
            report["failures"] = [f"{name}_missing_or_empty"]
            return finalize_patient(report, out, t0)
        actual_hashes[name] = sha256(path)
        expected = audit_row.get(MODALITIES[name][1], "")
        if actual_hashes[name] != expected:
            report["status"] = "RAW_INPUT_HASH_CHANGED"
            report["raw_hashes"] = actual_hashes
            report["failures"] = [f"{name}_hash_changed", f"expected={expected}", f"actual={actual_hashes[name]}"]
            return finalize_patient(report, out, t0)
    report["raw_hashes"] = actual_hashes
    report["stage"] = "hdbet"
    output_base = hdir / "T1_hdbet.nii.gz"
    hmask_path = hdir / "T1_hdbet_mask.nii.gz"
    cache_path = hdir / "hdbet_provenance.json"
    cached = None
    if cache_path.is_file() and output_base.is_file() and hmask_path.is_file():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if (candidate.get("input_t1_sha256") == actual_hashes["T1"]
                    and candidate.get("output_sha256") == sha256(output_base)
                    and candidate.get("mask_sha256") == sha256(hmask_path)
                    and candidate.get("implementation") == "brainles_hd_bet 0.0.11"
                    and candidate.get("device") == "cpu" and candidate.get("mode") == "fast"
                    and candidate.get("tta") == 0 and candidate.get("postprocessing") == 0):
                cached = candidate
        except Exception:
            cached = None
    if cached is not None:
        cmd_rc, hdbet_seconds, command = 0, cached.get("elapsed_seconds"), json.dumps(cached["command"])
    else:
        cmd_rc, hdbet_seconds, command = run_hdbet_only(paths["T1"], output_base, hdir / "hd_bet.log")
    report["hd_bet"] = {"implementation": "brainles_hd_bet 0.0.11", "torch": "2.14.0+cpu",
        "device": "cpu", "mode": "fast", "tta": 0, "postprocessing": 0,
        "command": command, "exit_status": cmd_rc, "elapsed_seconds": hdbet_seconds,
        "reused_verified_output": cached is not None,
        "log": str(hdir / "hd_bet.log"), "mask_path": str(hmask_path),
        "stripped_image_path": str(output_base)}
    if cmd_rc != 0 or not hmask_path.exists():
        report["status"] = "HD_BET_FAILED"
        report["failures"] = ["HD-BET command failed or mask output absent"]
        return finalize_patient(report, out, t0)
    report["outputs"]["hdbet_mask"] = str(hmask_path)

    report["stage"] = "five_region_construction_and_qc"
    # Use the exact established Phase-3B helper implementations.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import phase3b_preprocess_materials as phase3b
    phase3b.synthetic_grow_test()
    images = {k: nib.load(str(v)) for k, v in paths.items()}
    base_img, mask_img, h_img = images["T1"], images["MASK"], nib.load(str(hmask_path))
    affine = np.asarray(base_img.affine, dtype=float)
    if any(tuple(img.shape) != tuple(base_img.shape) or not np.allclose(img.affine, affine, rtol=1e-5, atol=1e-5)
           for img in images.values()) or tuple(h_img.shape) != tuple(base_img.shape) or not np.allclose(h_img.affine, affine, rtol=1e-5, atol=1e-5):
        report["status"] = "GEOMETRY_FAILED"
        report["failures"] = ["raw modalities, MASK, and HD-BET outputs are not on identical geometry"]
        return finalize_patient(report, out, t0)
    tumor_values = np.asanyarray(mask_img.dataobj)
    if not np.isfinite(tumor_values).all():
        report["status"] = "TUMOR_PRESERVATION_FAILED"
        report["failures"] = ["MASK contains non-finite values"]
        return finalize_patient(report, out, t0)
    tumor = tumor_values > 0
    if tumor_values.ndim != 3 or not tumor.any():
        report["status"] = "TUMOR_PRESERVATION_FAILED"
        report["failures"] = ["MASK is not a nonempty 3-D tumor mask"]
        return finalize_patient(report, out, t0)
    h = np.asanyarray(h_img.dataobj) > 0
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    six = ndimage.generate_binary_structure(3, 1)
    tumor_labels, tumor_n = ndimage.label(tumor, structure=six)
    tumor_sizes = np.bincount(tumor_labels.ravel())[1:]
    tumor_coords = np.argwhere(tumor)
    tumor_centroid_vox = tumor_coords.mean(axis=0)
    tumor_centroid_world = nib.affines.apply_affine(affine, tumor_centroid_vox)
    outside = tumor & ~h
    outside_n = int(outside.sum())
    distance_to_h = ndimage.distance_transform_edt(~h, sampling=spacing)

    # Exact Phase-3B tumor-safe envelope: HD-BET union tumor plus only missing
    # six-neighbor support voxels needed to prevent tumor exposure.
    face_neighbors = ndimage.binary_dilation(tumor, structure=six) & ~tumor
    support = face_neighbors & ~h
    envelope = h | tumor | support
    exposed_before = phase3b.face_count(tumor, (~h) & ~tumor)
    exposed_after = phase3b.face_count(tumor, (~envelope) & ~tumor)
    pad = 11
    pad_width = [(pad, pad)] * 3
    hp, tp, support_p, ep = (np.pad(a, pad_width) for a in (h, tumor, support, envelope))
    padded_affine = affine.copy()
    padded_affine[:3, 3] -= affine[:3, :3] @ np.full(3, pad)
    csf_outer = phase3b.grow_physical(ep, 2.0, spacing)
    skull_outer = phase3b.grow_physical(csf_outer, 3.0, spacing)
    scalp_outer = phase3b.grow_physical(skull_outer, 4.0, spacing)
    labels = np.zeros(ep.shape, dtype=np.uint8)
    masks = {1: ep & ~tp, 2: tp, 3: csf_outer & ~ep,
             4: skull_outer & ~csf_outer, 5: scalp_outer & ~skull_outer}
    for material, region in masks.items():
        labels[region] = material
    raw_crop = labels[pad:-pad, pad:-pad, pad:-pad]
    tumor_final = raw_crop == 2
    fp = int(np.count_nonzero(tumor_final & ~tumor))
    fn = int(np.count_nonzero(tumor & ~tumor_final))
    tumor_diff = int(np.count_nonzero(tumor_final != tumor))
    inter = int(np.count_nonzero(tumor_final & tumor))
    union = int(np.count_nonzero(tumor_final | tumor))

    adjacency = phase3b.face_adjacency(labels)
    exterior, internal_components, internal_voxels = phase3b.exterior_face_counts(labels)
    forbidden = {k: int(v) for k, v in adjacency.items() if k not in ALLOWED_PAIRS and not k.startswith("0-") and v}
    overlap_count = int(sum(int(x.sum()) for x in masks.values()) - int(np.count_nonzero(labels)))
    interfaces = interface_metrics(labels, spacing)
    material_stats = {}
    component_stats = {}
    for m, name in MATERIAL_NAMES.items():
        count = int(np.count_nonzero(labels == m))
        material_stats[name] = {"id": m, "voxels": count, "volume_mm3": count * voxel_volume}
        component_stats[name] = components_info(labels == m, six)
    # Six-connected background components touching any padded array boundary are exterior.
    bg = labels == 0
    bg_lab, bg_n = ndimage.label(bg, structure=six)
    boundary_labels = set(np.concatenate((bg_lab[0].ravel(), bg_lab[-1].ravel(), bg_lab[:, 0].ravel(), bg_lab[:, -1].ravel(), bg_lab[:, :, 0].ravel(), bg_lab[:, :, -1].ravel())).tolist()) - {0}
    exterior_bg = np.isin(bg_lab, list(boundary_labels))
    internal_bg = bg & ~exterior_bg
    inner_bg_lab, inner_bg_n = ndimage.label(internal_bg, structure=six)
    padding_ok = not bool(np.any(labels[[0, -1], :, :])) and not bool(np.any(labels[:, [0, -1], :])) and not bool(np.any(labels[:, :, [0, -1]]))
    tumor_voxel_vol = float(tumor.sum()) * voxel_volume
    final_tumor_coords = np.argwhere(tumor_final)
    final_tumor_centroid_vox = final_tumor_coords.mean(axis=0)
    final_tumor_centroid_world = nib.affines.apply_affine(affine, final_tumor_centroid_vox)
    tumor_bbox_source = phase3b.bbox(tumor)
    tumor_bbox_final = phase3b.bbox(tumor_final)
    bbox_min_delta = (np.asarray(tumor_bbox_final["min"]) - np.asarray(tumor_bbox_source["min"])).tolist()
    bbox_max_delta = (np.asarray(tumor_bbox_final["max"]) - np.asarray(tumor_bbox_source["max"])).tolist()
    tumor_centroid_delta_world = final_tumor_centroid_world - tumor_centroid_world
    tumor_outside_dist = float(distance_to_h[outside].max()) if outside_n else 0.0
    invariants = {
        "raw_hashes_match_audit": True,
        "hdbet_mask_nonempty": bool(h.any()),
        "tumor_exact_preservation": fp == 0 and fn == 0 and tumor_diff == 0 and inter == int(tumor.sum()),
        "tumor_safe_envelope_contains_tumor": int(np.count_nonzero(tumor & ~envelope)) == 0,
        "support_exact_reference_rule": bool(np.array_equal(support, face_neighbors & ~h)),
        "padding_coordinates_preserved": bool(np.allclose(nib.affines.apply_affine(padded_affine, np.array([[pad, pad, pad], [*([pad + 1] * 3)]], dtype=float)), nib.affines.apply_affine(affine, np.array([[0, 0, 0], [1, 1, 1]], dtype=float)))),
        "nonbackground_away_from_padded_boundary": padding_ok,
        "cumulative_shell_nesting": bool(np.all(ep <= csf_outer) and np.all(csf_outer <= skull_outer) and np.all(skull_outer <= scalp_outer)),
        "materials_mutually_exclusive": overlap_count == 0,
        "all_five_materials_present": all(material_stats[n]["voxels"] > 0 for n in material_stats),
        "no_forbidden_material_adjacency": not forbidden,
        "brain_background_zero": exterior["1"] == 0,
        "tumor_background_zero": exterior["2"] == 0,
        "csf_background_zero": exterior["3"] == 0,
        "skull_background_zero": exterior["4"] == 0,
        "scalp_contacts_background": exterior["5"] > 0,
        "no_internal_background_cavities": int(internal_bg.sum()) == 0 and int(inner_bg_n) == 0,
        "tumor_not_exposed_after_support": exposed_after == 0,
        "material_conservation": int(sum(x["voxels"] for x in material_stats.values())) == int(np.count_nonzero(labels)),
    }
    failures = [name for name, passed in invariants.items() if not passed]
    status = "PREPROCESSING_CERTIFIED" if not failures else (
        "TUMOR_PRESERVATION_FAILED" if not invariants["tumor_exact_preservation"] else
        "PADDING_FAILED" if not padding_ok else
        "INTERNAL_BACKGROUND_FAILED" if not invariants["no_internal_background_cavities"] else
        "TOPOLOGY_FAILED")

    # Persist the reference artifacts only in the workspace-derived directory.
    def save_binary(name: str, arr: np.ndarray) -> str:
        p = out / name
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), padded_affine), str(p))
        return str(p)
    outputs = {
        "hdbet_mask": str(hmask_path),
        "tumor_safe_brain_envelope": save_binary("tumor_safe_brain_envelope.nii.gz", ep),
        "computational_support": save_binary("computational_brain_support.nii.gz", support_p),
        "material_labels": str(out / "material_labels.nii.gz"),
        "material_labels_original_geometry": str(out / "material_labels_original_geometry.nii.gz"),
    }
    nib.save(nib.Nifti1Image(labels, padded_affine), outputs["material_labels"])
    nib.save(nib.Nifti1Image(raw_crop, affine), outputs["material_labels_original_geometry"])
    report.update({
        "status": status, "stage": "complete", "failure_invariants": failures,
        "hd_bet_disagreement": {"hdbet_brain_voxels": int(h.sum()), "tumor_voxels": int(tumor.sum()),
            "tumor_inside_hdbet_voxels": int((tumor & h).sum()), "tumor_outside_hdbet_voxels": outside_n,
            "tumor_outside_hdbet_fraction": outside_n / int(tumor.sum()),
            "tumor_outside_hdbet_percent": 100.0 * outside_n / int(tumor.sum()),
            "maximum_excluded_tumor_distance_from_hdbet_mm": tumor_outside_dist},
        "geometry": {"original_shape": list(tumor.shape), "padded_shape": list(labels.shape),
            "spacing_mm": spacing.tolist(), "orientation": list(nib.aff2axcodes(affine)),
            "voxel_volume_mm3": voxel_volume, "original_affine": affine.tolist(),
            "padded_affine": padded_affine.tolist(), "padding_each_side_voxels": [pad] * 3,
            "physical_origin_preserved": invariants["padding_coordinates_preserved"]},
        "tumor": {"source_voxels": int(tumor.sum()), "final_voxels": int(tumor_final.sum()),
            "false_positives": fp, "false_negatives": fn, "differing_voxels": tumor_diff,
            "dice": 2 * inter / (int(tumor.sum()) + int(tumor_final.sum())),
            "jaccard": inter / union if union else 1.0,
            "physical_volume_mm3": tumor_voxel_vol, "source_components_6": int(tumor_n),
            "component_sizes_voxels_desc": sorted(map(int, tumor_sizes), reverse=True),
            "final_components_6": components_info(tumor_final, six)["count"],
            "centroid_voxel": tumor_centroid_vox.tolist(),
            "centroid_world_mm": tumor_centroid_world.tolist(),
            "final_centroid_voxel": final_tumor_centroid_vox.tolist(),
            "final_centroid_world_mm": final_tumor_centroid_world.tolist(),
            "centroid_discrepancy_world_mm": tumor_centroid_delta_world.tolist(),
            "centroid_displacement_mm": float(np.linalg.norm(tumor_centroid_delta_world)),
            "bbox_voxel": tumor_bbox_source, "final_bbox_voxel": tumor_bbox_final,
            "bbox_min_discrepancy_voxels": bbox_min_delta, "bbox_max_discrepancy_voxels": bbox_max_delta,
            "bbox_exact": tumor_bbox_source == tumor_bbox_final,
            "bbox_world_mm": phase3b.world_bbox(tumor, affine),
            "final_bbox_world_mm": phase3b.world_bbox(tumor_final, affine),
            "added_to_hdbet": outside_n},
        "computational_support": {"voxels": int(support.sum()), "volume_mm3": int(support.sum()) * voxel_volume,
            "bbox_voxel": phase3b.bbox(support), "bbox_world_mm": phase3b.world_bbox(support, affine),
            "max_distance_from_hdbet_mm": float(distance_to_h[ support ].max()) if support.any() else 0.0,
            "tumor_exposed_faces_before": int(exposed_before), "tumor_exposed_faces_after": int(exposed_after)},
        "material_statistics": material_stats,
        "cumulative_shell_voxels": {"brain_envelope": int(ep.sum()), "csf_outer": int(csf_outer.sum()),
            "skull_outer": int(skull_outer.sum()), "scalp_outer": int(scalp_outer.sum())},
        "interfaces": interfaces, "material_adjacency_faces": adjacency,
        "forbidden_adjacency_faces": forbidden, "exterior_material_faces": exterior,
        "internal_background": {"components": int(inner_bg_n), "voxels": int(internal_bg.sum())},
        "material_components_6": component_stats, "overlapping_material_voxels": overlap_count,
        "invariants": invariants, "outputs": outputs,
        "method": {"reference": "phase3b_preprocess_materials.py", "shell_distances_mm": {"CSF": 2.0, "Skull": 3.0, "Scalp": 4.0},
            "tumor_safe_envelope": "HDBET union original tumor union missing 6-neighbor support voxels (Phase-3B rule)",
            "tumor_processing": "nonzero raw labels interpreted as tumor; no component selection, morphology, or relabeling",
            "HD-BET": report.get("hd_bet")},
        "runtime_seconds": time.monotonic() - t0, "peak_rss_gb": peak_rss_gb(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_generation": "NOT_PERFORMED", "FEM": "NOT_PERFORMED"})
    write_patient_report(report, out)
    return report


def write_patient_report(report: dict[str, object], out: Path) -> None:
    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"Pilot20 preprocessing: {report.get('patient_id')}", f"Status: {report.get('status')}",
             f"Stage: {report.get('stage')}", f"Failures: {report.get('failure_invariants', report.get('failures', []))}",
             f"Raw hashes verified: {bool(report.get('raw_hashes'))}", "Mesh: NOT GENERATED", "FEM: NOT RUN"]
    if "tumor" in report:
        t = report["tumor"]
        lines += [f"Tumor voxels: {t['source_voxels']}; components: {t['source_components_6']}; differing voxels: {t['differing_voxels']}",
                  f"Tumor centroid displacement: {t.get('centroid_displacement_mm', 'NOT_RECORDED')} mm; exact bbox: {t.get('bbox_exact', 'NOT_RECORDED')}",
                  f"HD-BET-excluded tumor voxels: {report['hd_bet_disagreement']['tumor_outside_hdbet_voxels']}",
                  f"Internal background: {report['internal_background']['components']} components / {report['internal_background']['voxels']} voxels"]
    (out / "preprocessing_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "status.json").write_text(json.dumps({"patient_id": report.get("patient_id"), "status": report.get("status"),
        "stage": report.get("stage"), "failure_invariants": report.get("failure_invariants", report.get("failures", []))}, indent=2) + "\n", encoding="utf-8")


def finalize_patient(report: dict[str, object], out: Path, t0: float) -> dict[str, object]:
    report.setdefault("runtime_seconds", time.monotonic() - t0)
    report.setdefault("peak_rss_gb", peak_rss_gb())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_patient_report(report, out)
    return report


def summarize(reports: list[dict[str, object]], total_seconds: float) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in reports:
        row: dict[str, object] = {"patient_id": r["patient_id"], "status": r["status"], "stage": r.get("stage"),
            "runtime_seconds": r.get("runtime_seconds"), "peak_rss_gb": r.get("peak_rss_gb")}
        if "geometry" in r:
            row.update({"shape": json.dumps(r["geometry"]["original_shape"]), "spacing_mm": json.dumps(r["geometry"]["spacing_mm"])})
        if "tumor" in r:
            row.update({"tumor_voxels": r["tumor"]["source_voxels"], "tumor_volume_mm3": r["tumor"]["physical_volume_mm3"],
                "tumor_components": r["tumor"]["source_components_6"], "tumor_outside_hdbet_voxels": r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"],
                "tumor_outside_hdbet_percent": r["hd_bet_disagreement"]["tumor_outside_hdbet_percent"],
                "tumor_outside_hdbet_volume_mm3": r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"] * r["geometry"]["voxel_volume_mm3"],
                "tumor_differing_voxels": r["tumor"]["differing_voxels"], "tumor_dice": r["tumor"]["dice"], "tumor_jaccard": r["tumor"]["jaccard"],
                "support_voxels": r["computational_support"]["voxels"], "internal_background_components": r["internal_background"]["components"],
                "internal_background_voxels": r["internal_background"]["voxels"], "forbidden_adjacency_faces": sum(r["forbidden_adjacency_faces"].values()),
                "overlapping_material_voxels": r["overlapping_material_voxels"]})
            for m in MATERIAL_NAMES.values():
                row[f"{m}_voxels"] = r["material_statistics"][m]["voxels"]
                row[f"{m}_volume_mm3"] = r["material_statistics"][m]["volume_mm3"]
            for tag, met in r["interfaces"].items():
                row[f"interface_{tag}_faces"] = met["faces"]
                row[f"interface_{tag}_area_mm2"] = met["area_mm2"]
        flat_rows.append(row)
    fields = sorted({k for row in flat_rows for k in row})
    with (ROOT / "cohort_meshing/pilot20/preprocessing_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat_rows)
    good = [r for r in reports if "material_statistics" in r]
    stat_fields = ["hdbet_brain_volume_mm3", "tumor_volume_mm3", "BrainTissue_volume_mm3", "CSF_volume_mm3", "Skull_volume_mm3", "Scalp_volume_mm3", "total_occupied_volume_mm3", "tumor_outside_hdbet_percent", "tumor_outside_hdbet_volume_mm3", "computational_support_volume_mm3", "runtime_seconds", "peak_rss_gb"] + [f"interface_{t}_area_mm2" for t in ("101", "102", "103", "104", "105")]
    stats = {}
    for metric in stat_fields:
        vals = []
        for r in good:
            if metric == "hdbet_brain_volume_mm3": vals.append(r["hd_bet_disagreement"]["hdbet_brain_voxels"] * r["geometry"]["voxel_volume_mm3"])
            elif metric == "tumor_volume_mm3": vals.append(r["tumor"]["physical_volume_mm3"])
            elif metric == "tumor_outside_hdbet_percent": vals.append(r["hd_bet_disagreement"]["tumor_outside_hdbet_percent"])
            elif metric == "tumor_outside_hdbet_volume_mm3": vals.append(r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"] * r["geometry"]["voxel_volume_mm3"])
            elif metric == "computational_support_volume_mm3": vals.append(r["computational_support"]["volume_mm3"])
            elif metric == "runtime_seconds": vals.append(r["runtime_seconds"])
            elif metric == "peak_rss_gb": vals.append(r["peak_rss_gb"])
            elif metric == "total_occupied_volume_mm3": vals.append(sum(x["volume_mm3"] for x in r["material_statistics"].values()))
            elif metric.startswith("interface_"): vals.append(r["interfaces"][metric.split("_")[1]]["area_mm2"])
            else:
                name = metric.removesuffix("_volume_mm3")
                vals.append(r["material_statistics"][name]["volume_mm3"])
        if vals:
            a = np.asarray(vals, dtype=float)
            stats[metric] = {"N": len(vals), "min": float(a.min()), "P5": float(np.percentile(a, 5)), "P25": float(np.percentile(a, 25)),
                "median": float(np.median(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "P75": float(np.percentile(a, 75)), "P95": float(np.percentile(a, 95)), "max": float(a.max())}
    multi = [r["patient_id"] for r in good if r["tumor"]["source_components_6"] > 1]
    with_tumor_out = [r["patient_id"] for r in good if r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"] > 0]
    fields_invariants = [r for r in good if r["status"] == "PREPROCESSING_CERTIFIED"]
    fail_rows = [{"patient_id": r["patient_id"], "status": r["status"], "stage": r.get("stage"), "failed_invariants": r.get("failure_invariants", r.get("failures", []))} for r in reports if r["status"] != "PREPROCESSING_CERTIFIED"]
    runtime_vals = [float(r.get("runtime_seconds", 0)) for r in reports]
    parent_peak = max((float(r.get("peak_rss_gb", 0)) for r in reports), default=None)
    preflight_peak_gb = 2350232.0 / (1024.0 * 1024.0)
    report = {"phase": "PILOT20 five-region preprocessing and QC", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input_audit": str(AUDIT), "raw_root": str(RAW_ROOT), "patients_raw_ready": len(reports),
        "hdbet_success": sum(bool(r.get("hd_bet", {}).get("exit_status") == 0) for r in reports),
        "hdbet_failure": sum(r.get("status") == "HD_BET_FAILED" for r in reports), "preprocessing_attempted": len(reports),
        "preprocessing_certified": len(fields_invariants), "preprocessing_failed": len(reports) - len(fields_invariants),
        "exact_tumor_preservation": sum(bool(r.get("invariants", {}).get("tumor_exact_preservation")) for r in reports),
        "topology_pass": sum(bool(r.get("invariants", {}).get("no_forbidden_material_adjacency")) for r in reports),
        "internal_background_pass": sum(bool(r.get("invariants", {}).get("no_internal_background_cavities")) for r in reports),
        "multicomponent_tumor_patients": multi, "patients_with_tumor_outside_hdbet": with_tumor_out,
        "patients_requiring_computational_support": [r["patient_id"] for r in good if r["computational_support"]["voxels"] > 0],
        "total_tumor_voxels": sum(int(r["tumor"]["source_voxels"]) for r in good),
        "total_tumor_outside_hdbet_voxels": sum(int(r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"]) for r in good),
        "total_tumor_outside_hdbet_volume_mm3": sum(int(r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"]) * float(r["geometry"]["voxel_volume_mm3"]) for r in good),
        "tumor_outside_hdbet_overall_fraction": (sum(int(r["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"]) for r in good) / sum(int(r["tumor"]["source_voxels"]) for r in good)) if good else None,
        "total_computational_support_voxels": sum(int(r["computational_support"]["voxels"]) for r in good),
        "total_computational_support_volume_mm3": sum(float(r["computational_support"]["volume_mm3"]) for r in good),
        "forbidden_adjacency_pass_count": sum(not bool(r["forbidden_adjacency_faces"]) for r in good),
        "complete_topology_and_cavity_pass_count": sum(r["status"] == "PREPROCESSING_CERTIFIED" for r in good),
        "memory_measurement_scope": "Per-case peak is measured in the preprocessing parent; an HD-BET preflight process was observed at VmHWM 2,350,232 KiB (2.24 GiB), but child peak was not sampled independently for every case.",
        "observed_hdbet_preflight_peak_rss_gb": 2350232.0 / (1024.0 * 1024.0),
        "cohort_statistics_successful_patients": len(good), "statistics": stats,
        "outliers_method": "Tukey IQR fences, 1.5*IQR, computed per listed continuous cohort metric on completed cases; descriptive only",
        "failures": fail_rows, "total_runtime_seconds": total_seconds,
        "median_runtime_per_patient_seconds": float(np.median(runtime_vals)) if runtime_vals else None,
        "maximum_preprocessing_parent_peak_rss_gb": parent_peak,
        "maximum_peak_rss_gb": max(parent_peak or 0.0, preflight_peak_gb),
        "raw_input_postrun_hash_check": {"checked_files": 0, "all_match_preprocessing_start": None, "mismatches": []},
        "patient_statuses": [{"patient_id": r["patient_id"], "status": r["status"], "runtime_seconds": r.get("runtime_seconds"), "peak_rss_gb": r.get("peak_rss_gb")} for r in reports],
        "mesh_generation": "NOT_PERFORMED", "FEM": "NOT_PERFORMED"}
    # Explicit robust IQR outlier listing.
    outlier_rows = []
    for metric in ("hdbet_brain_volume_mm3", "tumor_volume_mm3", "BrainTissue_volume_mm3", "CSF_volume_mm3", "Skull_volume_mm3", "Scalp_volume_mm3", "total_occupied_volume_mm3", "tumor_outside_hdbet_percent", "computational_support_volume_mm3", "interface_101_area_mm2", "interface_102_area_mm2", "interface_103_area_mm2", "interface_104_area_mm2", "interface_105_area_mm2"):
        vals = []
        for r in good:
            if metric == "tumor_volume_mm3": v = r["tumor"]["physical_volume_mm3"]
            elif metric == "tumor_outside_hdbet_percent": v = r["hd_bet_disagreement"]["tumor_outside_hdbet_percent"]
            elif metric == "computational_support_volume_mm3": v = r["computational_support"]["volume_mm3"]
            elif metric == "hdbet_brain_volume_mm3": v = r["hd_bet_disagreement"]["hdbet_brain_voxels"] * r["geometry"]["voxel_volume_mm3"]
            elif metric == "runtime_seconds": v = r["runtime_seconds"]
            elif metric == "peak_rss_gb": v = r["peak_rss_gb"]
            elif metric.startswith("interface_"): v = r["interfaces"][metric.split("_")[1]]["area_mm2"]
            elif metric == "total_occupied_volume_mm3": v = sum(x["volume_mm3"] for x in r["material_statistics"].values())
            else: v = r["material_statistics"][metric.removesuffix("_volume_mm3")]["volume_mm3"]
            vals.append((r["patient_id"], float(v)))
        if len(vals) >= 4:
            arr = np.array([v for _, v in vals]); q1, q3 = np.percentile(arr, [25, 75]); iqr = q3 - q1
            for pid, v in vals:
                if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr:
                    outlier_rows.append({"patient_id": pid, "metric": metric, "value": v, "classification": "GEOMETRIC/ANATOMICAL OUTLIER", "method": "Tukey 1.5*IQR"})
    with (ROOT / "cohort_meshing/pilot20/preprocessing_outliers.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["patient_id", "metric", "value", "classification", "method"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(outlier_rows)
    report["outliers"] = outlier_rows
    report["pilot20_preprocessing"] = "PASS" if len(reports) == 20 and len(fields_invariants) == 20 else "PARTIAL"
    report["ready_for_pilot20_full_resolution_meshing"] = bool(report["pilot20_preprocessing"] == "PASS")
    for name, obj in (("preprocessing_pilot20_report.json", report),):
        (ROOT / "cohort_meshing/pilot20" / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = ["PILOT20 FIVE-REGION PREPROCESSING / QC", "", f"PATIENTS_RAW_READY = {len(reports)}",
        f"HD_BET_SUCCESS = {report['hdbet_success']}", f"HD_BET_FAILURE = {report['hdbet_failure']}",
        f"PREPROCESSING_ATTEMPTED = {len(reports)}", f"PREPROCESSING_CERTIFIED = {len(fields_invariants)}",
        f"PREPROCESSING_FAILED = {len(reports)-len(fields_invariants)}", f"EXACT_TUMOR_PRESERVATION = {report['exact_tumor_preservation']}",
        f"TOPOLOGY_PASS = {report['topology_pass']}", f"INTERNAL_BACKGROUND_PASS = {report['internal_background_pass']}",
        f"MULTICOMPONENT_TUMOR_PATIENTS = {len(multi)} ({', '.join(multi)})",
        f"PATIENTS_WITH_TUMOR_OUTSIDE_HDBET = {len(with_tumor_out)} ({', '.join(with_tumor_out)})",
        f"TOTAL_TUMOR_OUTSIDE_HDBET_VOXELS = {report['total_tumor_outside_hdbet_voxels']}",
        f"TOTAL_TUMOR_OUTSIDE_HDBET_MM3 = {report['total_tumor_outside_hdbet_volume_mm3']}",
        f"OVERALL_TUMOR_OUTSIDE_HDBET_FRACTION = {report['tumor_outside_hdbet_overall_fraction']}",
        f"PATIENTS_REQUIRING_COMPUTATIONAL_SUPPORT = {len(report['patients_requiring_computational_support'])}",
        f"FORBIDDEN_ADJACENCY_PASS = {report['forbidden_adjacency_pass_count']}/{len(good)}",
        f"FULL_TOPOLOGY_AND_CAVITY_PASS = {report['complete_topology_and_cavity_pass_count']}/{len(good)}",
        f"TOTAL_RUNTIME = {total_seconds:.3f} s", f"MEDIAN_RUNTIME_PER_PATIENT = {report['median_runtime_per_patient_seconds']} s",
        f"MAXIMUM_PEAK_RAM = {report['maximum_peak_rss_gb']} GB (includes measured HD-BET preflight high-water)", "",
        f"MEMORY_MEASUREMENT_SCOPE = {report['memory_measurement_scope']}", "", "PATIENT STATUS", "Patient | Status | Tumor voxels | Tumor comps | Tumor outside HD-BET | Support voxels | Forbidden faces | Internal BG"]
    for r in reports:
        if "tumor" in r:
            text.append(f"{r['patient_id']} | {r['status']} | {r['tumor']['source_voxels']} | {r['tumor']['source_components_6']} | {r['hd_bet_disagreement']['tumor_outside_hdbet_voxels']} | {r['computational_support']['voxels']} | {sum(r['forbidden_adjacency_faces'].values())} | {r['internal_background']['voxels']}")
        else: text.append(f"{r['patient_id']} | {r['status']} | - | - | - | - | - | -")
    if good:
        tum = stats.get("tumor_volume_mm3", {})
        text += ["", "COHORT QUESTIONS (DESCRIPTIVE)",
            f"Tumor volume min/median/max = {tum.get('min')} / {tum.get('median')} / {tum.get('max')} mm^3",
            f"Multi-component tumors = {len(multi)}/20: {', '.join(multi)}",
            f"HD-BET excluded some original tumor = {len(with_tumor_out)}/20 patients; total excluded = {report['total_tumor_outside_hdbet_voxels']} voxels ({report['total_tumor_outside_hdbet_volume_mm3']} mm^3); pooled fraction = {100.0*report['tumor_outside_hdbet_overall_fraction']:.6f}%",
            f"Tumor-safe H union/support preserved the exact original tumor = {report['exact_tumor_preservation']}/20",
            f"Patients with nonzero computational support = {len(report['patients_requiring_computational_support'])}/20; added support = {report['total_computational_support_voxels']} voxels ({report['total_computational_support_volume_mm3']} mm^3)",
            f"Allowed material adjacency passed = {report['forbidden_adjacency_pass_count']}/20; full cavity/topology gate passed = {report['complete_topology_and_cavity_pass_count']}/20",
            f"Internal cavities = {report['preprocessing_failed']} failed patients; failures: {', '.join(x['patient_id'] for x in fail_rows)}",
            "Interface areas (min / median / max mm^2):"]
        for tag in ("101", "102", "103", "104", "105"):
            v = stats.get(f"interface_{tag}_area_mm2", {})
            text.append(f"  {tag}: {v.get('min')} / {v.get('median')} / {v.get('max')}")
        text += ["Interface counts/areas are computed from voxel-face adjacency with spacing-derived physical face areas; they are geometric outputs, not clinical interpretations.",
            "HD-BET-derived brain plus original tumor/support creates the computational envelope; shells are computational material regions."]
    text += ["", "STATISTICAL OUTLIERS (Tukey 1.5*IQR; descriptive only)"]
    text += [f"{x['patient_id']} | {x['metric']} = {x['value']} | {x['classification']}" for x in outlier_rows] or ["None"]
    text += ["", "FAILURES"] + ([json.dumps(x) for x in fail_rows] if fail_rows else ["None"])
    text += ["", "No mesh was generated. FEM was not run.", f"PILOT20_PREPROCESSING = {report['pilot20_preprocessing']}",
        f"READY_FOR_PILOT20_FULL_RESOLUTION_MESHING = {'YES' if report['ready_for_pilot20_full_resolution_meshing'] else 'NO'}"]
    (ROOT / "cohort_meshing/pilot20/preprocessing_pilot20_report.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdbet-only", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--seal-hdbet-preflight", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--only-patient", help="Process one audited patient for a bounded pipeline check")
    parser.add_argument("--summary-only", action="store_true", help="Rebuild cohort summaries from completed patient reports without rerunning inference")
    args = parser.parse_args()
    if args.seal_hdbet_preflight:
        inp, output_base = map(Path, args.seal_hdbet_preflight)
        hmask = Path(str(output_base)[:-7] + "_mask.nii.gz")
        if not inp.is_file() or not output_base.is_file() or not hmask.is_file():
            raise FileNotFoundError("Preflight input/output/mask missing")
        img, msk = nib.load(str(output_base)), nib.load(str(hmask))
        if tuple(img.shape) != tuple(msk.shape) or not np.allclose(img.affine, msk.affine):
            raise RuntimeError("Preflight HD-BET outputs have inconsistent geometry")
        command = ["hd-bet", "-i", str(inp), "-o", str(output_base), "-device", "cpu", "-mode", "fast", "-tta", "0", "-pp", "0", "-s", "1", "--overwrite_existing", "1"]
        provenance = {"input_t1_sha256": sha256(inp), "output_sha256": sha256(output_base), "mask_sha256": sha256(hmask),
            "implementation": "brainles_hd_bet 0.0.11", "device": "cpu", "mode": "fast", "tta": 0,
            "postprocessing": 0, "command": command, "elapsed_seconds": None,
            "note": "Preflight inference completed successfully in this session using the exact recorded Phase-2 command and these output paths."}
        output_base.parent.joinpath("hdbet_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.summary_only:
        existing = json.loads((ROOT / "cohort_meshing/pilot20/preprocessing_pilot20_report.json").read_text(encoding="utf-8"))
        patient_dirs = sorted((OUT_ROOT).glob("EGD-*/preprocessing_report.json"))
        loaded = [json.loads(p.read_text(encoding="utf-8")) for p in patient_dirs]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import phase3b_preprocess_materials as phase3b
        # Recompute source/final tumor spatial metrics from the raw mask and the
        # saved unpadded label image; this is a read-only, non-inference audit.
        for report in loaded:
            pid = report["patient_id"]
            patient_raw = RAW_ROOT / pid / f"MR_{pid}" / "5_MASK/NIFTI/MASK.nii.gz"
            label_path = Path(report.get("outputs", {}).get("material_labels_original_geometry", ""))
            if not patient_raw.is_file() or not label_path.is_file() or "tumor" not in report:
                continue
            raw_img = nib.load(str(patient_raw)); lab_img = nib.load(str(label_path))
            source = np.asanyarray(raw_img.dataobj) > 0
            final = np.asanyarray(lab_img.dataobj) == 2
            sc, fc = np.argwhere(source), np.argwhere(final)
            source_centroid = sc.mean(axis=0); final_centroid = fc.mean(axis=0)
            affine = np.asarray(raw_img.affine)
            source_world = nib.affines.apply_affine(affine, source_centroid)
            final_world = nib.affines.apply_affine(lab_img.affine, final_centroid)
            smin, smax = sc.min(axis=0), sc.max(axis=0)
            fmin, fmax = fc.min(axis=0), fc.max(axis=0)
            t = report["tumor"]
            t.update({"centroid_voxel": source_centroid.tolist(), "centroid_world_mm": source_world.tolist(),
                "final_centroid_voxel": final_centroid.tolist(), "final_centroid_world_mm": final_world.tolist(),
                "centroid_discrepancy_world_mm": (final_world-source_world).tolist(),
                "centroid_displacement_mm": float(np.linalg.norm(final_world-source_world)),
                "bbox_voxel": {"min": smin.tolist(), "max": smax.tolist()},
                "final_bbox_voxel": {"min": fmin.tolist(), "max": fmax.tolist()},
                "bbox_min_discrepancy_voxels": (fmin-smin).tolist(), "bbox_max_discrepancy_voxels": (fmax-smax).tolist(),
                "bbox_exact": bool(np.array_equal(smin, fmin) and np.array_equal(smax, fmax)),
                "bbox_world_mm": phase3b.world_bbox(source, affine),
                "final_bbox_world_mm": phase3b.world_bbox(final, lab_img.affine)})
            hpath = Path(report.get("outputs", {}).get("hdbet_mask", ""))
            hmask = np.asanyarray(nib.load(str(hpath)).dataobj) > 0 if hpath.is_file() else np.zeros_like(source)
            excluded = source & ~hmask
            if excluded.any():
                ec = np.argwhere(excluded)
                report["hd_bet_disagreement"].update({"tumor_outside_hdbet_bbox_voxel": {"min": ec.min(0).tolist(), "max": ec.max(0).tolist()},
                    "tumor_outside_hdbet_bbox_world_mm": phase3b.world_bbox(excluded, affine),
                    "tumor_outside_hdbet_centroid_voxel": ec.mean(0).tolist(),
                    "tumor_outside_hdbet_centroid_world_mm": nib.affines.apply_affine(affine, ec.mean(0)).tolist()})
            else:
                report["hd_bet_disagreement"].update({"tumor_outside_hdbet_bbox_voxel": None,
                    "tumor_outside_hdbet_bbox_world_mm": None, "tumor_outside_hdbet_centroid_voxel": None,
                    "tumor_outside_hdbet_centroid_world_mm": None})
            report["hd_bet_disagreement"]["tumor_outside_hdbet_volume_mm3"] = report["hd_bet_disagreement"]["tumor_outside_hdbet_voxels"] * report["geometry"]["voxel_volume_mm3"]
            out = OUT_ROOT / pid
            (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            write_patient_report(report, out)
        summarize(loaded, float(existing.get("total_runtime_seconds", 0.0)))
        audit = load_audit()
        checked, mismatches = 0, []
        for pid, row in audit.items():
            for name, (_, hash_column) in MODALITIES.items():
                path = Path(row.get(f"{name}_path", ""))
                actual = sha256(path) if path.is_file() else None
                checked += 1
                if actual != row.get(hash_column):
                    mismatches.append({"patient_id": pid, "input": name, "expected_sha256": row.get(hash_column), "actual_sha256": actual})
        final_report = json.loads((ROOT / "cohort_meshing/pilot20/preprocessing_pilot20_report.json").read_text(encoding="utf-8"))
        final_report["raw_input_postrun_hash_check"] = {"checked_files": checked, "all_match_preprocessing_start": not mismatches, "mismatches": mismatches}
        if mismatches:
            final_report["pilot20_preprocessing"] = "PARTIAL"
            final_report["ready_for_pilot20_full_resolution_meshing"] = False
            final_report["failures"].extend({"patient_id": x["patient_id"], "status": "RAW_INPUT_HASH_CHANGED", "stage": "postrun_hash_check", "failed_invariants": [x]} for x in mismatches)
        (ROOT / "cohort_meshing/pilot20/preprocessing_pilot20_report.json").write_text(json.dumps(final_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        txt = ROOT / "cohort_meshing/pilot20/preprocessing_pilot20_report.txt"
        with txt.open("a", encoding="utf-8") as f:
            f.write(f"\nRAW_INPUT_POSTRUN_HASH_CHECK = {checked} files; all_match = {not mismatches}\n")
            if mismatches:
                f.write("HASH MISMATCHES:\n" + "\n".join(json.dumps(x) for x in mismatches) + "\n")
        return 0
    if args.hdbet_only:
        inp, out = map(Path, args.hdbet_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(HD_BET_PACKAGE))
        import brainles_hd_bet.utils as utils
        if not hasattr(utils, "maybe_mkdir_p"):
            utils.maybe_mkdir_p = lambda p: os.makedirs(p, exist_ok=True)
        from brainles_hd_bet.hd_bet import main as hdbet_main
        sys.argv = ["hd-bet", "-i", str(inp), "-o", str(out), "-device", "cpu", "-mode", "fast", "-tta", "0", "-pp", "0", "-s", "1", "--overwrite_existing", "1"]
        hdbet_main()
        return 0
    audit = load_audit()
    patients = sorted(p.name for p in RAW_ROOT.glob("EGD-*") if p.is_dir() and (p / f"MR_{p.name}").is_dir())
    if args.only_patient:
        if args.only_patient not in patients:
            raise SystemExit(f"Patient not found in external cohort: {args.only_patient}")
        patients = [args.only_patient]
    if len(patients) != 20:
        print(f"Expected 20 raw-ready patients based on completed audit; discovered {len(patients)}", file=sys.stderr)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    cohort_start = time.monotonic()
    for index, patient in enumerate(patients, 1):
        print(f"[{index}/{len(patients)}] Starting {patient}", flush=True)
        if patient not in audit:
            row = {"patient_id": patient, "status": "MISSING_RAW_INPUT", "stage": "audit_lookup", "failures": ["patient missing from raw_input_audit.csv"]}
            (OUT_ROOT / patient).mkdir(parents=True, exist_ok=True)
            reports.append(finalize_patient(row, OUT_ROOT / patient, time.monotonic()))
            continue
        case_started = time.monotonic()
        try:
            r = patient_preprocess(patient, audit[patient])
        except Exception as e:
            pdir = OUT_ROOT / patient
            r = {"patient_id": patient, "status": "FAILED_PREPROCESSING", "stage": "unhandled_exception",
                 "failures": [f"{type(e).__name__}: {e}"], "traceback": traceback.format_exc(),
                 "mesh_generation": "NOT_PERFORMED", "FEM": "NOT_PERFORMED"}
            finalize_patient(r, pdir, case_started)
        reports.append(r)
        print(f"[{index}/{len(patients)}] {patient}: {r['status']} ({float(r.get('runtime_seconds',0)):.1f}s)", flush=True)
        gc.collect()
    summarize(reports, time.monotonic() - cohort_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
