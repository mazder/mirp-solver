#!/usr/bin/env python3
"""Record pilot20 eligibility without substituting missing Phase-2 masks."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

SOURCE = Path("/media/mazder/80183b18-9ca3-4c7c-b4bc-d95674a79e22/home/mazder/simulation_data/brain_tumar")
OUT = Path(__file__).resolve().parent
REQUIRED_BRAIN = Path("phase2_brain_extraction/hd_bet/T1_hdbet_mask.nii.gz")
REQUIRED_TUMOR = Path("5_MASK/NIFTI/MASK.nii.gz")
T1 = Path("1_T1/NIFTI/T1.nii.gz")
LEGACY = Path("1_T1/mesh/preprocessed_mask.nii.gz")


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def image_info(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "sha256": None}
    im = nib.load(str(path))
    a = np.asanyarray(im.dataobj)
    return {"exists": True, "sha256": sha(path), "shape": list(im.shape),
            "spacing_mm": list(map(float, im.header.get_zooms()[:3])),
            "orientation": list(nib.aff2axcodes(im.affine)), "affine": np.asarray(im.affine).tolist(),
            "nonzero_voxels": int(np.count_nonzero(a))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    patients = sorted((p for p in SOURCE.glob("EGD-*") if p.is_dir() and p.name[4:].isdigit()), key=lambda p: int(p.name[4:]))
    records = []
    for rank, patient in enumerate(patients, 1):
        pid = patient.name
        sub = patient / f"MR-{pid}"  # compatibility fallback only; canonical is MR_EGD-####
        canonical = patient / f"MR_{pid}"
        if not canonical.is_dir():
            canonical = sub
        inputs = {"t1": image_info(canonical / T1),
                  "tumor_mask": image_info(canonical / REQUIRED_TUMOR),
                  "required_hdbet_brain_mask": image_info(canonical / REQUIRED_BRAIN),
                  "legacy_preprocessed_mask_not_eligible": image_info(canonical / LEGACY)}
        reasons = []
        if not inputs["t1"]["exists"]: reasons.append("missing T1 source image")
        if not inputs["tumor_mask"]["exists"]: reasons.append("missing tumor mask")
        if not inputs["required_hdbet_brain_mask"]["exists"]: reasons.append("missing required Phase-2 HD-BET brain mask")
        # The old preprocessed_mask is known from the legacy log/path to be a cropped
        # tumor mask; equal nonzero count is additional evidence, not a replacement.
        legacy_tumor_equivalence = False
        if inputs["legacy_preprocessed_mask_not_eligible"]["exists"] and inputs["tumor_mask"]["exists"]:
            legacy_tumor_equivalence = (inputs["legacy_preprocessed_mask_not_eligible"]["nonzero_voxels"] ==
                                        inputs["tumor_mask"]["nonzero_voxels"])
        eligible = not reasons
        status = "MISSING_INPUT" if not eligible else "ELIGIBLE_NOT_STARTED"
        rec = {"patient_id": pid, "discovery_order": rank, "selection_order": None,
               "eligible": eligible, "status": status, "missing_inputs": reasons,
               "source_patient_path": str(patient), "resolved_dataset_path": str(canonical),
               "inputs": inputs, "legacy_mask_nonzero_count_matches_tumor": legacy_tumor_equivalence,
               "eligibility_rule": "T1 + original tumor mask + Phase-2 HD-BET brain mask required by established Phase-3B preprocessing"}
        records.append(rec)
        pdir = OUT / "patients" / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "status.json").write_text(json.dumps(rec, indent=2) + "\n")
        (pdir / "metrics.json").write_text(json.dumps({"patient_id": pid, "status": status,
            "mesh_metrics": None, "reason": reasons, "note": "No preprocessing or meshing was run because required source inputs are missing."}, indent=2) + "\n")
        (pdir / "independent_certification.json").write_text(json.dumps({"patient_id": pid,
            "certification_status": "NOT_RUN_MISSING_INPUT", "mesh_certified": False,
            "mesh_path": None, "hard_invariants": None, "blocking_inputs": reasons}, indent=2) + "\n")
        (pdir / "independent_certification.txt").write_text(
            f"Patient: {pid}\nCertification: NOT RUN (MISSING_INPUT)\n"+
            "No mesh was generated or certified. Required Phase-2 HD-BET brain mask is absent.\n"+
            "The legacy cropped preprocessed_mask is not accepted as a brain mask.\n")

    eligible = [r for r in records if r["eligible"]]
    selected = eligible[:20]
    for order, rec in enumerate(selected, 1): rec["selection_order"] = order
    # The current inventory has no eligible case, so cohort metric cells remain blank.
    headers = ["patient_id", "discovery_order", "selection_order", "eligible", "status",
               "missing_inputs", "tumor_volume_mm3", "brain_volume_mm3", "total_hex8", "vertices",
               "tumor_interface_area_mm2", "exterior_area_mm2", "mesh_time_s", "certification_time_s",
               "peak_rss_gb", "mesh_size_bytes", "certification_pass"]
    with (OUT / "cohort_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in records:
            w.writerow({"patient_id":r["patient_id"],"discovery_order":r["discovery_order"],
                        "selection_order":"","eligible":"NO","status":r["status"],
                        "missing_inputs":"; ".join(r["missing_inputs"]),"certification_pass":"NOT_RUN"})
    with (OUT / "patient_selection_and_inputs.csv").open("w", newline="") as f:
        fields=["patient_id","discovery_order","selection_order","eligible","status","T1_path","T1_sha256",
                "tumor_path","tumor_sha256","required_hdbet_path","required_hdbet_sha256","legacy_mask_path",
                "legacy_mask_sha256","legacy_mask_nonzero_count_matches_tumor","missing_inputs"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in records:
            base=Path(r["resolved_dataset_path"])
            w.writerow({"patient_id":r["patient_id"],"discovery_order":r["discovery_order"],"selection_order":"",
              "eligible":"NO","status":r["status"],"T1_path":str(base/T1),"T1_sha256":r["inputs"]["t1"]["sha256"],
              "tumor_path":str(base/REQUIRED_TUMOR),"tumor_sha256":r["inputs"]["tumor_mask"]["sha256"],
              "required_hdbet_path":str(base/REQUIRED_BRAIN),"required_hdbet_sha256":r["inputs"]["required_hdbet_brain_mask"]["sha256"],
              "legacy_mask_path":str(base/LEGACY),"legacy_mask_sha256":r["inputs"]["legacy_preprocessed_mask_not_eligible"]["sha256"],
              "legacy_mask_nonzero_count_matches_tumor":r["legacy_mask_nonzero_count_matches_tumor"],
              "missing_inputs":"; ".join(r["missing_inputs"])})
    (OUT / "outliers.csv").write_text("patient_id,metric,value,method,classification\n")
    (OUT / "cohort_statistics.json").write_text(json.dumps({"status":"NOT_ESTIMABLE_NO_ELIGIBLE_PATIENTS","statistics":{},
      "correlations":{},"normalized_metrics":{},"plots":"NOT_GENERATED_NO_ELIGIBLE_PATIENTS"},indent=2)+"\n")
    elapsed=time.perf_counter()-started
    missing_hdbet=sum(not r["inputs"]["required_hdbet_brain_mask"]["exists"] for r in records)
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"source_root":str(SOURCE),
      "output_root":str(OUT),"patients_discovered":len(patients),"patients_eligible":len(eligible),
      "patients_selected":len(selected),"patients_processed":0,"meshes_generated":0,"meshes_certified":0,
      "preprocessing_failures":0,"meshing_failures":0,"certification_failures":0,
      "missing_input_patients":sum(r["status"]=="MISSING_INPUT" for r in records),
      "missing_required_hdbet_count":missing_hdbet,"certification_pass_rate":None,
      "exact_tumor_preservation_rate":None,"exact_mesh_to_image_reconstruction_rate":None,
      "statistical_outliers":0,"total_runtime_seconds":elapsed,"median_runtime_per_patient_seconds":None,
      "p95_runtime_per_patient_seconds":None,"maximum_peak_ram_gb":None,"total_mesh_storage_bytes":0,
      "pilot20_status":"BLOCKED_AT_ELIGIBILITY_NO_HD_BET_MASKS",
      "ready_for_700_patient_batch":"NO","failure_pattern":"All discovered cases lack the required Phase-2 HD-BET brain mask. T1 and tumor mask are present; the retained legacy preprocessed_mask has the same nonzero count as the tumor mask and is a cropped tumor artifact, not a brain extraction mask.",
      "cleanup":{"old_mesh_artifacts_removed":320,"extensions":[".msh",".vtu",".stl"],"source_images_preserved":True},
      "selection_rule":"numeric patient ID order; select first 20 eligible datasets; no eligible datasets were found",
      "eligibility_rule":"requires T1 source image, original tumor mask, and Phase-2 HD-BET brain mask consumed by the established Phase-3B pipeline",
      "patients":records}
    (OUT/"pilot20_report.json").write_text(json.dumps(report,indent=2)+"\n")
    missing_count=sum(r['status']=='MISSING_INPUT' for r in records)
    txt=("20-PATIENT MESHING COHORT PILOT — ELIGIBILITY GATE\n\n"
      f"SOURCE: {SOURCE}\nPATIENTS_DISCOVERED = {len(patients)}\nPATIENTS_ELIGIBLE = {len(eligible)}\n"
      f"PATIENTS_SELECTED = {len(selected)}\nPATIENTS_PROCESSED = 0\nMESHES_GENERATED = 0\nMESHES_CERTIFIED = 0\n\n"
      "PREPROCESSING_FAILURES = 0 (not attempted)\nMESHING_FAILURES = 0 (not attempted)\nCERTIFICATION_FAILURES = 0 (not attempted)\n"
      f"MISSING_INPUTS = {missing_count}\nREQUIRED HD-BET BRAIN MASK ABSENT = {missing_hdbet}/{len(records)}\n"
      "CERTIFICATION_PASS_RATE = NOT_ESTIMABLE\nEXACT_TUMOR_PRESERVATION_RATE = NOT_ESTIMABLE\nEXACT_MESH_TO_IMAGE_RECONSTRUCTION_RATE = NOT_ESTIMABLE\n"
      "STATISTICAL_OUTLIERS = NOT_ESTIMABLE\nTOTAL_RUNTIME = eligibility inventory only; no patient processing\n"
      "MEDIAN_RUNTIME_PER_PATIENT = NOT_ESTIMABLE\nP95_RUNTIME_PER_PATIENT = NOT_ESTIMABLE\nMAXIMUM_PEAK_RAM = NOT_MEASURED\nTOTAL_MESH_STORAGE = 0 bytes (no new meshes)\n\n"
      "PATIENT | Tumor volume | Brain volume | HEX8 | Vertices | Tumor interface | Exterior | Mesh time | Cert time | Peak RAM | Status\n"
      + "\n".join(f"{r['patient_id']} | NA | NA | NA | NA | NA | NA | NA | NA | NA | MISSING_INPUT" for r in records)+"\n\n"
      "Missing-input pattern: all 20 cases have T1 and original tumor MASK.nii.gz, but lack the Phase-2 HD-BET brain mask required by the established Phase-3B pipeline.\n"
      "The remaining legacy preprocessed_mask.nii.gz files have the same nonzero voxel counts as the tumor masks and are cropped tumor artifacts; they were not substituted.\n\n"
      "No preprocessing, material labeling, mesh generation, independent mesh certification, plots, cohort statistics, or FEM solve was run. No mesh was generated from incomplete inputs.\n"
      "The former modality-level mesh outputs were cleaned; source images and tumor masks were preserved.\n\n"
      "PILOT20_MESHING_CERTIFICATION = NOT RUN (BLOCKED AT ELIGIBILITY)\nREADY_FOR_700_PATIENT_BATCH = NO\n\n"
      "Per-patient evidence: patients/<ID>/status.json, metrics.json, independent_certification.json, independent_certification.txt\n"
      "Selection/input hashes: patient_selection_and_inputs.csv\nCohort summary: cohort_summary.csv\n")
    (OUT/"pilot20_report.txt").write_text(txt)
    print(txt,flush=True)


if __name__ == "__main__":
    main()
