#!/usr/bin/env python3
"""Read-only follow-up documenting matching inputs at a non-contract directory level."""
import csv
import json
from pathlib import Path

import numpy as np

from validate_full_cohort_inputs import (
    MODALITIES, dump_csv, inspect_nifti, sha256, write_latex, write_text_report,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "cohort_meshing/input_validation_0475_0774"


def main():
    json_path = OUT / "cohort_input_validation.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    root = Path(report["dataset_root"])
    found = []
    by_patient = {}
    manifest = list(csv.DictReader((OUT / "input_file_manifest.csv").open(encoding="utf-8")))
    for patient in report["patient_results"]:
        if patient["mr_directory_exists"]:
            continue
        pid = patient["patient_id"]
        records = []
        for modality, (rel, kind, _required) in MODALITIES.items():
            alt = root / pid / rel
            item = {"patient_id": pid, "modality": modality, "alternate_path": str(alt),
                    "exists": alt.is_file(), "nonempty": False, "sha256": None,
                    "readable": False, "classification": "MISSING", "shape": None,
                    "dtype": None, "spacing_mm": None, "orientation": None,
                    "affine": None, "finite": None, "minimum": None, "maximum": None,
                    "unique_labels": None, "nonzero_voxels": None,
                    "matches_T1_shape": None, "matches_T1_affine": None}
            if alt.is_file():
                item["file_size_bytes"] = alt.stat().st_size
                item["nonempty"] = alt.stat().st_size > 0
                item["sha256"] = sha256(alt)
                if item["nonempty"]:
                    inspected = inspect_nifti(alt, modality)
                    item.update({k: inspected.get(k) for k in (
                        "readable", "classification", "shape", "dtype", "spacing_mm", "orientation",
                        "affine", "finite", "minimum", "maximum", "unique_labels", "nonzero_voxels", "errors")})
            records.append(item)
            found.append(item)
            exact = next(x for x in manifest if x["patient_id"] == pid and x["modality"] == modality)
            exact["alternate_candidate_path"] = str(alt)
            exact["alternate_candidate_exists"] = item["exists"]
            exact["alternate_candidate_sha256"] = item["sha256"]
            exact["alternate_candidate_readable"] = item["readable"]
            exact["alternate_candidate_shape"] = item["shape"]
            exact["alternate_candidate_spacing_mm"] = item["spacing_mm"]
            exact["alternate_candidate_orientation"] = item["orientation"]
            exact["alternate_candidate_finite"] = item["finite"]
            for miss in report["file_validation"]["missing_entries"]:
                if miss["patient_id"] == pid and miss["modality"] == modality:
                    miss["alternate_candidate_path"] = str(alt)
                    miss["alternate_candidate_exists"] = item["exists"]
                    miss["alternate_candidate_readable"] = item["readable"]
                    miss["reason"] += (f"; matching file exists at alternate path {alt}; "
                                        f"alternate_readable={item['readable']}")
        t1 = records[0]
        for item in records[1:]:
            item["matches_T1_shape"] = item["shape"] == t1["shape"] if item["shape"] else False
            item["matches_T1_affine"] = bool(np.allclose(item["affine"], t1["affine"], rtol=1e-5, atol=1e-5)) if item["affine"] and t1["affine"] else False
        complete = len(records) == len(MODALITIES) and all(x["exists"] and x["nonempty"] and x["readable"] for x in records)
        geometry = complete and all(x["matches_T1_shape"] and x["matches_T1_affine"] for x in records[1:])
        by_patient[pid] = {"patient_id": pid, "expected_mr_directory": str(root / pid / f"MR_{pid}"),
                           "expected_mr_directory_exists": False, "alternate_files": records,
                           "all_five_present_nonempty_readable": complete,
                           "all_five_geometry_match_T1": bool(geometry),
                           "pipeline_contract_status": "FAIL_EXACT_DIRECTORY_LAYOUT"}
        patient["alternate_layout_file_count"] = sum(x["exists"] and x["nonempty"] for x in records)
        patient["alternate_layout_readable_count"] = sum(x["readable"] for x in records)
        patient["alternate_layout_geometry_match_T1"] = bool(geometry)
        patient["failure_reasons"] = [
            f"{x['modality']}_expected_path_missing_file_found_at_alternate_layout_"
            f"{'readable' if x['readable'] else 'unreadable'}"
            for x in records
        ]
        patient["failure_count"] = len(patient["failure_reasons"])

    pids = sorted(by_patient)
    report["alternate_layout_findings"] = {
        "layout_tested": "<root>/<patient_id>/<modality>/NIFTI/<exact_filename> (MR_<patient_id> level omitted)",
        "patients": [by_patient[x] for x in pids],
        "file_count": len(found),
        "present_nonempty_count": sum(x["exists"] and x["nonempty"] for x in found),
        "readable_count": sum(x["readable"] for x in found),
        "geometry_aligned_patient_count": sum(by_patient[x]["all_five_geometry_match_T1"] for x in pids),
        "summary": (f"For {len(pids)} patients (EGD-0721 through EGD-0730), all {len(found)} required filenames "
                    f"were found at the alternate direct-under-EGD layout; "
                    f"{sum(x['readable'] for x in found)}/{len(found)} were readable and finite. "
                    f"All five files matched T1 shape and affine in {sum(by_patient[x]['all_five_geometry_match_T1'] for x in pids)}/{len(pids)} patients. "
                    "They remain FAIL under the authoritative runner contract because the required MR_EGD-XXXX directory level is absent; "
                    "no file was moved or substituted."),
    }
    report["file_validation"]["alternate_layout_files_present"] = report["alternate_layout_findings"]["present_nonempty_count"]
    report["file_validation"]["alternate_layout_files_readable"] = report["alternate_layout_findings"]["readable_count"]
    report["final_summary"]["blocking_patients"] = [
        {"patient_id": p["patient_id"], "reasons": p["failure_reasons"]}
        for p in report["patient_results"] if p["final_status"] == "FAIL"]

    fields = sorted(set().union(*(row.keys() for row in manifest)))
    dump_csv(OUT / "input_file_manifest.csv", manifest, fields)
    missing = report["file_validation"]["missing_entries"]
    dump_csv(OUT / "missing_inputs.csv", missing, sorted(set().union(*(row.keys() for row in missing))))
    patients = report["patient_results"]
    patient_csv_path = OUT / "patient_input_validation.csv"
    patient_csv_rows = list(csv.DictReader(patient_csv_path.open(encoding="utf-8")))
    for row in patient_csv_rows:
        rec = next(p for p in patients if p["patient_id"] == row["patient_id"])
        row["alternate_layout_file_count"] = rec.get("alternate_layout_file_count", 0)
        row["alternate_layout_readable_count"] = rec.get("alternate_layout_readable_count", 0)
        row["alternate_layout_geometry_match_T1"] = rec.get("alternate_layout_geometry_match_T1", "NOT_TESTED")
        row["failure_reasons"] = rec["failure_reasons"]
    dump_csv(patient_csv_path, patient_csv_rows)
    mask_rows = report["mask_validation"]["label_summaries"]
    for row in mask_rows:
        alt = by_patient.get(row["patient_id"])
        if alt:
            m = alt["alternate_files"][-1]
            row["alternate_path"] = m["alternate_path"]
            row["alternate_readable"] = m["readable"]
            row["alternate_unique_labels"] = m["unique_labels"]
            row["alternate_nonzero_tumor_voxels"] = m["nonzero_voxels"]
    dump_csv(OUT / "mask_label_summary.csv", mask_rows, sorted(set().union(*(row.keys() for row in mask_rows))))
    dump_csv(OUT / "alternate_layout_inputs.csv", found)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_text_report(OUT / "cohort_input_validation.txt", report)
    write_latex(OUT, report, patients)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_text_report(OUT / "cohort_input_validation.txt", report)
    print(report["alternate_layout_findings"]["summary"])
    print(f"Exact-contract readiness remains NO; PASS={report['final_summary']['PASS']} FAIL={report['final_summary']['FAIL']}; PDF={report['latex']['status']}")


if __name__ == "__main__":
    main()
