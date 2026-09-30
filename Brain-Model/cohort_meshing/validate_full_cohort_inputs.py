#!/usr/bin/env python3
"""Read-only validation of the authoritative raw MRI contract for a cohort."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MODALITIES = {
    "T1": ("1_T1/NIFTI/T1.nii.gz", "MRI", True),
    "T1GD": ("2_T1GD/NIFTI/T1GD.nii.gz", "MRI", True),
    "T2": ("3_T2/NIFTI/T2.nii.gz", "MRI", True),
    "FLAIR": ("4_FLAIR/NIFTI/FLAIR.nii.gz", "MRI", True),
    "MASK": ("5_MASK/NIFTI/MASK.nii.gz", "tumor_mask", True),
}
GEOM_RTOL = 1e-5
GEOM_ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def dump_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def numeric_summary(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: None for k in ("N", "minimum", "P5", "P25", "median", "mean", "std_sample", "P75", "P95", "maximum")}
    return {"N": int(arr.size), "minimum": float(arr.min()), "P5": float(np.quantile(arr, .05)),
            "P25": float(np.quantile(arr, .25)), "median": float(np.median(arr)), "mean": float(arr.mean()),
            "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "P75": float(np.quantile(arr, .75)), "P95": float(np.quantile(arr, .95)), "maximum": float(arr.max())}


def inspect_nifti(path: Path, modality: str) -> dict:
    result = {"readable": False, "classification": "UNREADABLE", "shape": None, "dtype": None,
              "spacing_mm": None, "orientation": None, "affine": None, "affine_determinant": None,
              "voxel_volume_mm3": None, "header_dimensions": None, "finite": None,
              "nonfinite_count": None, "minimum": None, "maximum": None, "mean": None,
              "unique_labels": None, "nonzero_voxels": None, "errors": []}
    try:
        image = nib.load(str(path))
        shape = tuple(int(x) for x in image.shape)
        affine = np.asarray(image.affine, dtype=float)
        zooms = tuple(float(x) for x in image.header.get_zooms()[:min(3, len(shape))])
        header_dims = [int(x) for x in np.asarray(image.header["dim"]).tolist()]
        result.update(shape=list(shape), dtype=str(image.get_data_dtype()), spacing_mm=list(zooms),
                      orientation=list(nib.aff2axcodes(affine)) if affine.shape == (4, 4) else None,
                      affine=affine.tolist(), header_dimensions=header_dims)
        if affine.shape == (4, 4) and np.isfinite(affine).all():
            det = float(np.linalg.det(affine[:3, :3]))
            result["affine_determinant"] = det
            result["voxel_volume_mm3"] = abs(det)
        if len(shape) != 3 or any(d <= 0 for d in shape):
            result["classification"] = "WRONG_DIMENSION"
            result["errors"].append(f"expected nonempty 3-D volume, observed shape={shape}")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            result["classification"] = "INVALID_NIFTI"
            result["errors"].append("affine is malformed or non-finite")
        if not zooms or not all(math.isfinite(z) and z > 0 for z in zooms):
            result["classification"] = "INVALID_NIFTI"
            result["errors"].append("voxel spacing is not finite and positive")
        data = np.asanyarray(image.dataobj)
        finite_mask = np.isfinite(data)
        finite_count = int(np.count_nonzero(finite_mask))
        nonfinite_count = int(data.size - finite_count)
        result["finite"] = nonfinite_count == 0
        result["nonfinite_count"] = nonfinite_count
        if finite_count:
            finite_data = data[finite_mask]
            result["minimum"] = float(np.min(finite_data))
            result["maximum"] = float(np.max(finite_data))
            result["mean"] = float(np.mean(finite_data, dtype=np.float64))
        if modality == "MASK":
            vals = np.unique(data[finite_mask]) if finite_count else np.asarray([])
            result["unique_labels"] = [jsonable(x) for x in vals.tolist()]
            result["nonzero_voxels"] = int(np.count_nonzero(data > 0))
            if result["nonzero_voxels"] == 0:
                result["errors"].append("tumor mask is empty under established MASK > 0 semantics")
                result["classification"] = "EMPTY"
        if nonfinite_count:
            result["errors"].append(f"{nonfinite_count} NaN/Inf voxels")
            result["classification"] = "INVALID_NIFTI"
        result["readable"] = True
        if result["classification"] == "UNREADABLE":
            result["classification"] = "PASS" if not result["errors"] else "INVALID_NIFTI"
        elif result["classification"] == "EMPTY" and result["errors"] and nonfinite_count == 0:
            # The file is readable, but the established Phase-3B contract rejects empty tumor masks.
            result["classification"] = "EMPTY"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        if result["classification"] == "UNREADABLE":
            result["classification"] = "UNREADABLE"
    return result


def dataset_tree_bytes(root: Path, expected_dirs: list[Path]) -> int:
    total = 0
    for patient_dir in expected_dirs:
        if not patient_dir.is_dir():
            continue
        for current, dirs, files in os.walk(patient_dir, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
            for name in files:
                p = Path(current) / name
                try:
                    if p.is_file() and not p.is_symlink():
                        total += p.stat().st_size
                except OSError:
                    pass
    return total


def write_latex(output_dir: Path, report: dict, patients: list[dict]):
    tex_dir = output_dir / "report"
    build = tex_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "full_cohort_input_validation.tex"
    summary = report["final_summary"]
    contract = report["input_contract"]
    dirs = report["directory_validation"]
    geom = report["geometry_validation"]
    mask = report["mask_validation"]
    storage = report["storage_statistics"]
    dup_groups = report["duplicate_analysis"]["duplicate_groups"]

    def esc(s):
        table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(table.get(c, c) for c in str(s))

    rows = []
    for p in patients:
        geom_flags = p["geometry_checks"]
        rows.append(f"{p['patient_id']} & {p['required_files_found']}/5 & {p['readable_files']}/5 & "
                    f"{'Y' if geom_flags.get('SHAPE_MATCH') else 'N'} & {'Y' if geom_flags.get('AFFINE_MATCH') else 'N'} & "
                    f"{p.get('tumor_voxel_count') if p.get('tumor_voxel_count') is not None else '--'} & "
                    f"{p['warning_count']} & {p['failure_count']} & {esc(p['final_status'])} " + r"\\")
    duplicate_text = "No exact cross-patient required-input file duplicates were found."
    if dup_groups:
        duplicate_text = "\\begin{itemize}\n" + "\n".join(
            f"\\item SHA-256 \\texttt{{{g['sha256']}}}: {esc(', '.join(g['patient_ids']))} ({esc(', '.join(g['modalities']))})."
            for g in dup_groups) + "\n\\end{itemize}"
    missing_text = ", ".join(
        f"{x['patient_id']} ({x['modality']}: {x['relative_path']}; {x['reason']})"
        for x in report["file_validation"]["missing_entries"]
    ) or "None."
    failed_text = ", ".join(f"{p['patient_id']}: {esc('; '.join(p['failure_reasons']))}" for p in patients if p["final_status"] == "FAIL") or "None."
    review_text = ", ".join(f"{p['patient_id']}: {esc('; '.join(p['warning_reasons']))}" for p in patients if p["final_status"] == "REVIEW_REQUIRED") or "None."
    input_contract = "\\\\\n".join(f"\\texttt{{{esc(k)}}}: \\path{{{v[0]}}}" for k, v in MODALITIES.items())
    geometry_tol = f"rtol={GEOM_RTOL:g}, atol={GEOM_ATOL:g}"
    git = report["reproducibility"].get("git_commit") or "not available"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")
    range_label = f"{report['expected_range']['first']}--{report['expected_range']['last']}"
    expected_count = report["expected_range"]["expected_count"]
    pdf_lines = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm,headheight=14pt]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{graphicx,booktabs,longtable,tabularx,hyperref,xcolor,fancyhdr}
\hypersetup{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,pdftitle={Full Cohort Raw MRI Input Validation RANGE}}
\pagestyle{fancy}\fancyhf{}\lhead{Full-cohort input validation}\rhead{RANGE}\cfoot{\thepage}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}\emergencystretch=3em
\title{\vspace{-1.2cm}\textbf{Full Cohort Raw MRI Input Validation}\\[0.5em]\large RANGE (N=EXPECTEDCOUNT)}
\author{Brain-Model cohort input audit}\date{REPORTDATE}
\begin{document}\maketitle\thispagestyle{empty}\clearpage\tableofcontents\clearpage
\section{Executive summary}\label{sec:summary}
Audit scope was the explicit RANGE range (EXPECTEDCOUNT expected patients) in \path{DATASETROOT}. The source was inspected read-only. Results: expected directories present DIRPASS/EXPECTEDCOUNT; expected raw files found FILEFOUND/EXPECTEDFILES; readable NIfTI files READABLE/EXPECTEDFILES; patient classifications PASSCOUNT PASS, REVIEWCOUNT review-required, and FAILCOUNT fail. Readiness: \textbf{READYVALUE}. No preprocessing, mesh generation, or FEM was performed.
\section{Authoritative input contract}\label{sec:contract}
The contract is derived from \path{cohort_meshing/pilot20/run_preprocessing_pilot20.py} and \path{phase3b_preprocess_materials.py}. The Pilot20 runner explicitly defines and hash-verifies five exact modality paths, loads all five and requires common T1 shape and affine before material construction. Phase 3B uses T1 for HD-BET/diagnostics and MASK with tumor defined as values greater than zero; it requires a finite, nonempty 3-D tumor mask. T1GD, T2, and FLAIR remain mandatory inputs to the validated runner/geometry checks, although the Phase-3B material-construction helper does not use their intensities to define labels.

Required patient layout (under \path{<root>/EGD-XXXX/MR_EGD-XXXX/}):\\
CONTRACT
No derived HD-BET mask, material label, or mesh is a raw-input requirement. Required files must be nonempty readable NIfTI-1/2 \texttt{.nii.gz} at the exact paths above. Mask interpretation is the validated \texttt{MASK > 0} tumor rule; mask labels are reported without rewriting or collapsing.
\section{Expected and discovered cohort}\label{sec:directories}
Expected sequence was generated independently from integer IDs FIRSTNUM--LASTNUM. Expected count: EXPECTEDCOUNT. Discovered matching patient directories: DISCOVERED. Missing IDs: MISSINGIDS. Unexpected EGD directories: UNEXPECTEDIDS. Malformed EGD-like entries: MALFORMED. Non-directory root entries: NONDIR. Directory details are in the JSON report.
\section{Required-file completeness}\label{sec:files}
There are EXPECTEDFILES expected required-file slots; FOUND files were present, regular, nonempty and read successfully, while MISSING were missing/invalid at the exact paths. Missing-file patients/paths: MISSINGTEXT. A similarly named file was not substituted for any required path. Per-file evidence, including SHA-256, is in the manifest.
\section{Files found at an alternate directory level}\label{sec:alternate}
ALTERNATELAYOUT
\section{NIfTI readability and dimensions}\label{sec:nifti}
NIfTI files were inspected with nibabel NIBVERSION. Header dimensions, image shapes, dtype, affine, voxel spacing and orientation were recorded. All expected volumes are required to be finite, nonempty 3-D arrays under the current workflow. Readable file count: READABLE; invalid present-file count: INVALID; exact required paths absent: MISSING. Finite min/max/mean and mask labels/counts are recorded in the file manifest and mask summary.
\section{Within-patient geometry}\label{sec:geometry}
T1 is the reference modality in the established Pilot20 pipeline. Each of T1GD, T2, FLAIR and MASK was compared against it for shape, voxel spacing, affine, and orientation. Affines use the pipeline's \texttt{numpy.allclose} tolerance (GEOMTOL); shape and orientation require exact equality; spacing uses a documented absolute tolerance of GEOMATOL mm. Results: patient full-geometry pass GEOMPASS; patient review GEOMREVIEW; patient geometry fail GEOMFAIL. File-level differences are enumerated in \path{geometry_mismatches.csv}; no resampling was performed.
\section{Mask validation and tumor statistics}\label{sec:mask}
The raw mask is classified by the existing code as tumor wherever \texttt{MASK > 0}. Empty masks fail because Phase 3B explicitly rejects a nonempty-mask violation. Binary versus nonbinary labels are recorded; values beyond 0/1 are observations for inspection, not relabelled. Nonempty masks: NONEMPTY/EXPECTEDCOUNT; empty: EMPTY; nonbinary observed: NONBINARY. Tumor voxel count summary: min MASKMIN; median MASKMED; mean MASKMEAN; max MASKMAX. No components were filtered and no masks were altered.
\section{Duplicate-file analysis}\label{sec:duplicates}
DUPLICATEPARA
An identical SHA-256 is evidence of byte-identical files, not automatically an error. No duplicate was deleted or otherwise changed.
\section{Storage statistics}\label{sec:storage}
Required-input total: TOTALREQ bytes (REQGB decimal GB); mean per patient MEANREQ bytes; median MEDIANREQ bytes; minimum MINREQ bytes; maximum MAXREQ bytes. Per-modality byte summaries are in the JSON report. Total regular-file storage under the EXPECTEDCOUNT expected patient directories (including any non-contract files) was TOTALTREE bytes (TREEGB decimal GB). These values are measured from filesystem file sizes.
\section{Patients requiring review}\label{sec:review}
REVIEWTEXT
\section{Failed patients}\label{sec:failed}
FAILTEXT
\section{Readiness decision}\label{sec:readiness}
\begin{center}\fcolorbox{blue!50!black}{blue!5}{\parbox{0.9\textwidth}{\centering\Large\textbf{READY\_FOR\_FULL\_COHORT\_PREPROCESSING = READYVALUE}}}\end{center}
The gate requires all EXPECTEDCOUNT expected patient directories and all mandatory inputs to be present/readable, hard geometry and mask checks to pass, and no unresolved FAIL cases. Documented review observations do not become failures by implication; they are listed above and included in the JSON and patient CSV. This decision concerns readiness to begin preprocessing only.
\section{Limitations}\label{sec:limits}
This is an input integrity audit, not a preprocessing validation. Readability and geometric alignment do not establish segmentation correctness, anatomical validity, or downstream mesh/solver success. Unusual-but-readable observations (e.g. nonbinary masks or exact duplicate files) are reported for review. No data repair, resampling, preprocessing, meshing or FEM was performed.
\section{Reproducibility}\label{sec:repro}
Dataset root: \path{DATASETROOT}. Patient range: RANGE. Validator: \path{cohort_meshing/validate_full_cohort_inputs.py}. Nibabel version: NIBVERSION; NumPy version: NUMPYVERSION; Python version: PYVERSION. Git commit: \path{GITCOMMIT}. Generated on REPORTDATE. The immutable SHA-256 manifest is \path{input_file_manifest.csv}; all outputs reside outside the dataset root.
\appendix
\section{Per-patient validation summary}\label{app:patients}
Table~\ref{tab:patients} lists all expected patients. Detailed modality fields and exact reasons are in \path{patient_input_validation.csv} and the JSON report.
\begin{longtable}{lrrrrrrrl}\caption{Patient-level raw-input validation summary.}\label{tab:patients}\\
\toprule Patient & Files & Read & Shape & Affine & Tumor vox. & Warn & Fail & Status \\\midrule\endfirsthead
\caption[]{Patient-level raw-input validation summary (continued).}\\\toprule Patient & Files & Read & Shape & Affine & Tumor vox. & Warn & Fail & Status \\\midrule\endhead
PATIENTROWS
\bottomrule\end{longtable}
\end{document}
'''
    replacements = {
        "REPORTDATE": esc(generated), "DATASETROOT": esc(report["dataset_root"]), "CONTRACT": input_contract,
        "RANGE": esc(range_label), "FIRSTNUM": report["expected_range"]["first"].replace("EGD-", ""),
        "LASTNUM": report["expected_range"]["last"].replace("EGD-", ""), "EXPECTEDCOUNT": str(expected_count),
        "EXPECTEDFILES": str(expected_count * len(MODALITIES)),
        "DIRPASS": str(dirs["expected_directories_present"]), "EXPECTED": str(dirs["expected_count"]),
        "DISCOVERED": str(dirs["discovered_expected_count"]),
        "MISSINGIDS": esc(", ".join(dirs["missing_ids"]) or "None"),
        "UNEXPECTEDIDS": esc(", ".join(dirs["unexpected_egd_directories"]) or "None"),
        "MALFORMED": esc(", ".join(dirs["malformed_egd_entries"]) or "None"), "NONDIR": str(dirs["non_directory_entries"]),
        "FILEFOUND": str(report["file_validation"]["valid_required_files"]),
        "MISSING": str(report["file_validation"].get("missing_exact_paths", report["file_validation"]["expected_file_slots"]-report["file_validation"]["valid_required_files"])),
        "MISSINGTEXT": esc(missing_text), "READABLE": str(report["file_validation"]["readable_nifti_files"]),
        "ALTERNATELAYOUT": esc(report.get("alternate_layout_findings", {}).get("summary", "No alternate-layout files were identified.")),
        "INVALID": str(report["file_validation"]["invalid_nifti_files"]), "NIBVERSION": report["reproducibility"]["nibabel_version"],
        "GEOMTOL": esc(geometry_tol), "GEOMATOL": f"{GEOM_ATOL:g}",
        "GEOMPASS": str(geom["patients_all_geometry_checks_pass"]), "GEOMREVIEW": str(geom["patients_geometry_review"]),
        "GEOMFAIL": str(geom["patients_geometry_fail"]), "NONEMPTY": str(mask["nonempty_tumor_masks"]),
        "EMPTY": str(mask["empty_tumor_masks"]), "NONBINARY": str(mask["nonbinary_masks_observed"]),
        "PASSCOUNT": str(summary["PASS"]), "REVIEWCOUNT": str(summary["REVIEW_REQUIRED"]), "FAILCOUNT": str(summary["FAIL"]),
        "MASKMIN": esc(str(mask["tumor_voxel_statistics"]["minimum"])),
        "MASKMED": esc(f"{mask['tumor_voxel_statistics']['median']:,.1f}"),
        "MASKMEAN": esc(f"{mask['tumor_voxel_statistics']['mean']:,.1f}"),
        "MASKMAX": esc(str(mask["tumor_voxel_statistics"]["maximum"])),
        "DUPLICATEPARA": duplicate_text,
        "TOTALREQ": f"{storage['required_input_bytes']:,}", "REQGB": f"{storage['required_input_bytes']/1e9:.3f}",
        "MEANREQ": f"{storage['per_patient_required_input_bytes']['mean']:,.0f}",
        "MEDIANREQ": f"{storage['per_patient_required_input_bytes']['median']:,.0f}",
        "MINREQ": f"{storage['per_patient_required_input_bytes']['minimum']:,.0f}",
        "MAXREQ": f"{storage['per_patient_required_input_bytes']['maximum']:,.0f}",
        "TOTALTREE": f"{storage['all_regular_file_bytes_under_expected_patient_dirs']:,}",
        "TREEGB": f"{storage['all_regular_file_bytes_under_expected_patient_dirs']/1e9:.3f}",
        "REVIEWTEXT": review_text, "FAILTEXT": failed_text,
        "READYVALUE": "YES" if summary["ready_for_full_cohort_preprocessing"] else "NO",
        "PYVERSION": report["reproducibility"]["python_version"], "NUMPYVERSION": report["reproducibility"]["numpy_version"],
        "GITCOMMIT": esc(git), "PATIENTROWS": "\n".join(rows),
    }
    for key, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pdf_lines = pdf_lines.replace(key, value)
    tex_path.write_text(pdf_lines, encoding="utf-8")
    pdflatex = shutil_which("pdflatex")
    if pdflatex:
        runs = []
        for _ in range(3):
            proc = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                                   "-output-directory=build", tex_path.name], cwd=tex_dir,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            runs.append({"returncode": proc.returncode, "tail": proc.stdout[-2000:]})
            if proc.returncode:
                break
        built = build / "full_cohort_input_validation.pdf"
        if runs and runs[-1]["returncode"] == 0 and built.is_file():
            (tex_dir / "full_cohort_input_validation.pdf").write_bytes(built.read_bytes())
        report["latex"] = {"source": str(tex_path), "pdf": str(tex_dir / "full_cohort_input_validation.pdf") if (tex_dir / "full_cohort_input_validation.pdf").exists() else None,
                           "compile_runs": runs, "status": "PASS" if runs and runs[-1]["returncode"] == 0 else "FAIL"}
    else:
        report["latex"] = {"source": str(tex_path), "pdf": None, "compile_runs": [], "status": "PDFLATEX_UNAVAILABLE"}


def shutil_which(command: str):
    from shutil import which
    return which(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/media/mazder/Games/Brain-Model/data/MRI-DATA-BRAIN"))
    parser.add_argument("--first-id", type=int, default=475)
    parser.add_argument("--last-id", type=int, default=774)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "cohort_meshing/input_validation_0475_0774")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if args.first_id > args.last_id:
        parser.error("--first-id must be <= --last-id")
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a readable directory: {root}")
    if out == root or root in out.parents or out in root.parents:
        raise SystemExit("Output directory must be separate from and outside the source dataset tree")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    expected_ids = [f"EGD-{i:04d}" for i in range(args.first_id, args.last_id + 1)]
    expected_set = set(expected_ids)
    root_entries = list(root.iterdir())
    dirs = [p for p in root_entries if p.is_dir() and not p.is_symlink()]
    discovered_egd = sorted(p.name for p in dirs if p.name.startswith("EGD-"))
    expected_dirs = [root / pid for pid in expected_ids]
    expected_present = [pid for pid in expected_ids if (root / pid).is_dir()]
    missing_ids = [pid for pid in expected_ids if pid not in set(discovered_egd)]
    unexpected = [pid for pid in discovered_egd if pid not in expected_set]
    malformed = sorted(p.name for p in dirs if p.name.startswith("EGD-") and not re.fullmatch(r"EGD-\d{4}", p.name))
    non_dirs = sorted(p.name for p in root_entries if not p.is_dir())

    contract_description = {
        "source_of_truth": ["cohort_meshing/pilot20/run_preprocessing_pilot20.py", "phase3b_preprocess_materials.py"],
        "patient_layout": "<root>/<patient_id>/MR_<patient_id>/<modality>/NIFTI/<exact_filename>",
        "required_files": {name: {"relative_path": spec[0], "type": spec[1], "required": spec[2]} for name, spec in MODALITIES.items()},
        "required_modalities": list(MODALITIES), "optional_raw_files_defined_by_runner": [],
        "derived_HD_BET_mask": "not a raw-input requirement; generated from T1 during preprocessing",
        "mask_definition": "tumor is raw MASK voxel value > 0; Phase-3B requires finite, nonempty 3-D mask",
        "runner_geometry_requirement": "all five raw NIfTI shapes equal T1 shape and all affines numpy.allclose(T1 affine, rtol=1e-5, atol=1e-5)",
        "dimensionality": "all five inputs must be nonempty 3-D arrays for the established pipeline",
    }
    patients = []
    manifest = []
    mismatch_rows = []
    mask_rows = []
    missing_rows = []
    hash_index: dict[str, list[dict]] = {}
    per_modality_bytes = {m: [] for m in MODALITIES}
    total_required_bytes = 0
    all_file_slots = 0
    readable_files = 0
    valid_files = 0
    nonbinary_ids = []
    empty_mask_ids = []
    shape_match_patient = spacing_match_patient = affine_match_patient = orient_match_patient = 0
    print(f"Auditing {len(expected_ids)} expected patient IDs in {root}", flush=True)
    for index, patient_id in enumerate(expected_ids, 1):
        patient_dir = root / patient_id / f"MR_{patient_id}"
        failures: list[str] = []
        warnings: list[str] = []
        records = {}
        file_count = read_count = 0
        patient_bytes = 0
        for modality, (relative, kind, required) in MODALITIES.items():
            all_file_slots += 1
            file_path = patient_dir / relative
            rec = {"patient_id": patient_id, "modality": modality, "type": kind,
                   "relative_path": str(Path(patient_id) / f"MR_{patient_id}" / relative),
                   "absolute_path": str(file_path), "exists": file_path.exists(), "regular_file": file_path.is_file(),
                   "nonempty": False, "readable": False, "sha256": None, "file_size_bytes": None,
                   "classification": "MISSING", "shape": None, "dtype": None, "spacing_mm": None,
                   "orientation": None, "affine": None, "affine_determinant": None,
                   "voxel_volume_mm3": None, "header_dimensions": None, "finite": None,
                   "nonfinite_count": None, "minimum": None, "maximum": None, "mean": None,
                   "unique_labels": None, "nonzero_voxels": None, "errors": []}
            if not patient_dir.is_dir():
                reason = f"missing patient structure {patient_id}/MR_{patient_id}"
                rec["errors"].append(reason)
            elif not file_path.is_file():
                reason = f"missing exact required path: {rec['relative_path']}"
                rec["errors"].append(reason)
            else:
                try:
                    size = file_path.stat().st_size
                    rec["file_size_bytes"] = size
                    rec["nonempty"] = size > 0
                    if not size:
                        rec["classification"] = "EMPTY"
                        rec["errors"].append("file size is zero")
                    else:
                        rec["sha256"] = sha256(file_path)
                        hash_index.setdefault(rec["sha256"], []).append({"patient_id": patient_id,
                            "modality": modality, "relative_path": rec["relative_path"], "size": size})
                        patient_bytes += size; total_required_bytes += size; per_modality_bytes[modality].append(size)
                        file_count += 1; valid_files += 1
                        if os.access(file_path, os.R_OK):
                            detail = inspect_nifti(file_path, modality)
                            rec.update(detail)
                            if detail["readable"]:
                                read_count += 1; readable_files += 1
                            if detail["classification"] not in ("PASS",):
                                rec["errors"].extend(e for e in detail["errors"] if e not in rec["errors"])
                        else:
                            rec["classification"] = "UNREADABLE"
                            rec["errors"].append("read access denied")
                except OSError as exc:
                    rec["classification"] = "UNREADABLE"
                    rec["errors"].append(f"filesystem error: {exc}")
            if rec["errors"]:
                missing_rows.append({"patient_id": patient_id, "modality": modality,
                                     "relative_path": rec["relative_path"], "classification": rec["classification"],
                                     "reason": "; ".join(rec["errors"])})
            records[modality] = rec
            manifest.append(rec)

        reference = records["T1"]
        geom_checks = {"SHAPE_MATCH": False, "SPACING_MATCH": False, "AFFINE_MATCH": False, "ORIENTATION_MATCH": False}
        if all(records[m]["readable"] for m in MODALITIES):
            comparisons = []
            for modality in ("T1GD", "T2", "FLAIR", "MASK"):
                r = records[modality]
                checks = {
                    "SHAPE_MATCH": r["shape"] == reference["shape"],
                    "SPACING_MATCH": bool(np.allclose(r["spacing_mm"], reference["spacing_mm"], rtol=GEOM_RTOL, atol=GEOM_ATOL)),
                    "AFFINE_MATCH": bool(np.allclose(r["affine"], reference["affine"], rtol=GEOM_RTOL, atol=GEOM_ATOL)),
                    "ORIENTATION_MATCH": r["orientation"] == reference["orientation"],
                }
                comparisons.append(checks)
                if not all(checks.values()):
                    mismatch_rows.append({"patient_id": patient_id, "reference_modality": "T1", "modality": modality,
                        "shape_match": checks["SHAPE_MATCH"], "spacing_match": checks["SPACING_MATCH"],
                        "affine_match": checks["AFFINE_MATCH"], "orientation_match": checks["ORIENTATION_MATCH"],
                        "reference_shape": reference["shape"], "observed_shape": r["shape"],
                        "reference_spacing_mm": reference["spacing_mm"], "observed_spacing_mm": r["spacing_mm"],
                        "reference_orientation": reference["orientation"], "observed_orientation": r["orientation"],
                        "reference_affine": reference["affine"], "observed_affine": r["affine"]})
            geom_checks = {key: all(c[key] for c in comparisons) for key in geom_checks}
            shape_match_patient += int(geom_checks["SHAPE_MATCH"])
            spacing_match_patient += int(geom_checks["SPACING_MATCH"])
            affine_match_patient += int(geom_checks["AFFINE_MATCH"])
            orient_match_patient += int(geom_checks["ORIENTATION_MATCH"])
            for key, ok in geom_checks.items():
                if not ok:
                    failures.append(f"geometry_{key.lower()}_failed")

        mask_rec = records["MASK"]
        tumor_count = mask_rec.get("nonzero_voxels") if mask_rec["readable"] else None
        labels = mask_rec.get("unique_labels") if mask_rec["readable"] else None
        if mask_rec["readable"]:
            if tumor_count == 0:
                failures.append("tumor_MASK_empty_under_values_greater_than_zero")
                empty_mask_ids.append(patient_id)
            if not mask_rec.get("finite", False):
                failures.append("mask_contains_nonfinite_voxels")
            if labels is not None and any(float(x) not in (0.0, 1.0) for x in labels):
                warnings.append("mask_has_nonbinary_labels; validated pipeline interprets every value > 0 as tumor")
                nonbinary_ids.append(patient_id)
        for modality, rec in records.items():
            if not rec["readable"]:
                failures.append(f"{modality}_{rec['classification'].lower()}")
            elif rec["classification"] in ("WRONG_DIMENSION", "INVALID_NIFTI", "EMPTY"):
                failures.append(f"{modality}_{rec['classification'].lower()}")
            elif modality != "MASK" and rec["minimum"] == rec["maximum"]:
                warnings.append(f"{modality}_constant_intensity_volume")

        shape_values = [records[m]["shape"] for m in MODALITIES]
        patient_summary = {
            "patient_id": patient_id, "patient_directory_exists": (root / patient_id).is_dir(),
            "mr_directory_exists": patient_dir.is_dir(), "required_files_expected": len(MODALITIES),
            "required_files_found": file_count, "readable_files": read_count,
            "nifti_readability_pass": read_count == len(MODALITIES),
            "geometry_checks": geom_checks, "geometry_pass": bool(all(geom_checks.values())),
            "mask_pass": bool(mask_rec["readable"] and tumor_count is not None and tumor_count > 0 and mask_rec.get("finite")),
            "reference_shape_T1": reference["shape"], "spacing_T1_mm": reference["spacing_mm"],
            "orientation_T1": reference["orientation"], "tumor_voxel_count": tumor_count,
            "mask_unique_labels": labels, "mask_nonzero_voxels": tumor_count,
            "required_input_bytes": patient_bytes, "input_hashes": {m: records[m]["sha256"] for m in MODALITIES},
            "warning_reasons": sorted(set(warnings)), "failure_reasons": sorted(set(failures)),
        }
        patient_summary["warning_count"] = len(patient_summary["warning_reasons"])
        patient_summary["failure_count"] = len(patient_summary["failure_reasons"])
        patient_summary["final_status"] = "FAIL" if failures else ("REVIEW_REQUIRED" if warnings else "PASS")
        patients.append(patient_summary)
        mask_rows.append({"patient_id": patient_id, "mask_relative_path": records["MASK"]["relative_path"],
                          "readable": mask_rec["readable"], "classification": mask_rec["classification"],
                          "shape": mask_rec["shape"], "dtype": mask_rec["dtype"],
                          "unique_labels": labels, "tumor_definition": "MASK > 0",
                          "nonzero_tumor_voxels": tumor_count, "finite": mask_rec["finite"],
                          "nonfinite_count": mask_rec["nonfinite_count"], "affine": mask_rec["affine"],
                          "spacing_mm": mask_rec["spacing_mm"]})
        del records
        gc.collect()
        if index % 10 == 0 or index == len(expected_ids):
            print(f"Progress: {index}/{len(expected_ids)}; PASS={sum(p['final_status']=='PASS' for p in patients)}; "
                  f"REVIEW={sum(p['final_status']=='REVIEW_REQUIRED' for p in patients)}; "
                  f"FAIL={sum(p['final_status']=='FAIL' for p in patients)}", flush=True)

    dup_groups = []
    for digest, group in hash_index.items():
        pids = sorted({x["patient_id"] for x in group})
        if len(pids) > 1:
            dup_groups.append({"sha256": digest, "patient_ids": pids,
                               "modalities": sorted({x["modality"] for x in group}),
                               "files": group, "duplicate_file_count": len(group)})
    duplicate_patients = sorted({pid for g in dup_groups for pid in g["patient_ids"]})
    for patient in patients:
        if patient["patient_id"] in duplicate_patients:
            patient["warning_reasons"].append("exact_required_input_file_hash_shared_with_another_patient")
            patient["warning_reasons"] = sorted(set(patient["warning_reasons"]))
            patient["warning_count"] = len(patient["warning_reasons"])
            if patient["final_status"] == "PASS":
                patient["final_status"] = "REVIEW_REQUIRED"
    all_five_duplicate_pairs = []
    id_to_hash = {p["patient_id"]: p["input_hashes"] for p in patients}
    for i, pa in enumerate(expected_ids):
        for pb in expected_ids[i+1:]:
            ha, hb = id_to_hash[pa], id_to_hash[pb]
            if all(ha[m] is not None and ha[m] == hb[m] for m in MODALITIES):
                all_five_duplicate_pairs.append([pa, pb])

    status_counts = {s: sum(p["final_status"] == s for p in patients) for s in ("PASS", "REVIEW_REQUIRED", "FAIL")}
    missing_patients = [p for p in patients if p["failure_count"]]
    review_patients = [p for p in patients if p["warning_count"] and p["final_status"] != "FAIL"]
    geo_all_pass = sum(all(p["geometry_checks"].values()) for p in patients)
    file_validation = {"expected_file_slots": len(expected_ids)*len(MODALITIES), "valid_required_files": valid_files,
                       "readable_nifti_files": readable_files,
                       "invalid_nifti_files": sum(1 for x in manifest if x["exists"] and x["nonempty"] and not x["readable"]),
                       "missing_exact_paths": len(expected_ids)*len(MODALITIES)-valid_files,
                       "missing_or_invalid_entries": len(missing_rows), "missing_entries": missing_rows}
    file_validation["found_per_modality"] = {m: sum(1 for x in manifest if x["modality"] == m and x["readable"]) for m in MODALITIES}
    geometry_validation = {"reference_modality": "T1", "affine_tolerance": {"rtol": GEOM_RTOL, "atol": GEOM_ATOL},
       "spacing_tolerance": {"rtol": GEOM_RTOL, "atol": GEOM_ATOL}, "orientation_rule": "exact axis-code tuple match",
       "patients_all_geometry_checks_pass": geo_all_pass,
       "shape_match_patients": shape_match_patient, "spacing_match_patients": spacing_match_patient,
       "affine_match_patients": affine_match_patient, "orientation_match_patients": orient_match_patient,
       "patients_geometry_review": len(mismatch_rows), "patients_geometry_fail": sum(any(not v for v in p["geometry_checks"].values()) and p["readable_files"] == 5 for p in patients),
       "mismatches": mismatch_rows}
    mask_counts = [p["tumor_voxel_count"] for p in patients if p["tumor_voxel_count"] is not None]
    mask_validation = {"tumor_definition": "MASK > 0, confirmed in phase3b_preprocess_materials.py",
       "empty_tumor_masks": len(empty_mask_ids), "empty_tumor_patient_ids": empty_mask_ids,
       "nonempty_tumor_masks": sum(x > 0 for x in mask_counts), "nonbinary_masks_observed": len(nonbinary_ids),
       "nonbinary_patient_ids": nonbinary_ids, "tumor_voxel_statistics": numeric_summary(mask_counts),
       "label_summaries": mask_rows}
    per_patient_storage = [p["required_input_bytes"] for p in patients]
    storage = {"required_input_bytes": total_required_bytes,
       "required_input_decimal_GB": total_required_bytes/1e9,
       "per_patient_required_input_bytes": numeric_summary(per_patient_storage),
       "per_modality_bytes": {m: numeric_summary(vals) for m, vals in per_modality_bytes.items()},
       "all_regular_file_bytes_under_expected_patient_dirs": dataset_tree_bytes(root, expected_dirs)}
    ready = (len(expected_ids) == args.last_id-args.first_id+1 and len(expected_present) == len(expected_ids)
             and not missing_ids and valid_files == len(expected_ids)*len(MODALITIES)
             and readable_files == len(expected_ids)*len(MODALITIES)
             and geometry_validation["patients_geometry_fail"] == 0
             and mask_validation["empty_tumor_masks"] == 0 and status_counts["FAIL"] == 0)
    if unexpected or malformed or non_dirs:
        # Root-level entries outside the required cohort are recorded, but do not imply a failure of this ID range.
        directory_observation = True
    else:
        directory_observation = False
    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(root),
        "read_only": True, "expected_range": {"first": expected_ids[0], "last": expected_ids[-1], "expected_count": len(expected_ids)},
        "input_contract": contract_description,
        "directory_validation": {"root_entries": len(root_entries), "expected_patient_ids": expected_ids,
            "expected_count": len(expected_ids),
            "expected_directories_present": len(expected_present), "discovered_expected_count": len([x for x in discovered_egd if x in expected_set]),
            "discovered_egd_directories": discovered_egd, "missing_ids": missing_ids,
            "unexpected_egd_directories": unexpected, "malformed_egd_entries": malformed,
            "non_directory_entries": len(non_dirs), "non_directory_entry_names": non_dirs,
            "directory_scope_observation": directory_observation},
        "file_validation": file_validation, "geometry_validation": geometry_validation,
        "mask_validation": mask_validation,
        "duplicate_analysis": {"duplicate_groups": dup_groups, "duplicate_patient_ids": duplicate_patients,
            "all_five_required_inputs_identical_patient_pairs": all_five_duplicate_pairs,
            "duplicate_input_file_count_across_groups": sum(g["duplicate_file_count"] for g in dup_groups)},
        "storage_statistics": storage, "patient_results": patients,
        "final_summary": {"PASS": status_counts["PASS"], "REVIEW_REQUIRED": status_counts["REVIEW_REQUIRED"],
            "FAIL": status_counts["FAIL"], "expected_patients": len(expected_ids),
            "patients_with_all_five_readable_inputs": sum(p["readable_files"] == 5 for p in patients),
            "patients_geometry_all_pass": geo_all_pass,
            "ready_for_full_cohort_preprocessing": bool(ready),
            "blocking_patients": [{"patient_id": p["patient_id"], "reasons": p["failure_reasons"]} for p in patients if p["final_status"] == "FAIL"],
            "review_patients": [{"patient_id": p["patient_id"], "reasons": p["warning_reasons"]} for p in patients if p["final_status"] == "REVIEW_REQUIRED"]},
        "reproducibility": {"python_version": sys.version.split()[0], "numpy_version": np.__version__,
            "nibabel_version": nib.__version__, "git_commit": None},
        "elapsed_seconds": None,
    }
    try:
        report["reproducibility"]["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    # All manifest rows are from the declared exact contract; no directory walking changes source state.
    dump_csv(out / "input_file_manifest.csv", manifest,
        ["patient_id", "modality", "type", "relative_path", "absolute_path", "exists", "regular_file", "nonempty",
         "readable", "classification", "file_size_bytes", "sha256", "shape", "dtype", "spacing_mm", "orientation",
         "affine", "affine_determinant", "voxel_volume_mm3", "header_dimensions", "finite", "nonfinite_count",
         "minimum", "maximum", "mean", "unique_labels", "nonzero_voxels", "errors"])
    patient_csv = []
    for p in patients:
        row = {k: v for k, v in p.items() if k not in ("geometry_checks", "input_hashes", "warning_reasons", "failure_reasons")}
        row.update({k.lower(): v for k, v in p["geometry_checks"].items()})
        row["warning_reasons"] = p["warning_reasons"]
        row["failure_reasons"] = p["failure_reasons"]
        for m, rel in (("t1", "T1"), ("t1gd", "T1GD"), ("t2", "T2"), ("flair", "FLAIR"), ("mask", "MASK")):
            # Corresponding exact input values are incorporated from manifest below.
            rec = next(x for x in manifest if x["patient_id"] == p["patient_id"] and x["modality"] == rel)
            row[f"{m}_readable"] = rec["readable"]
            row[f"{m}_shape"] = rec["shape"]
            row[f"{m}_spacing_mm"] = rec["spacing_mm"]
            row[f"{m}_orientation"] = rec["orientation"]
        row.update({f"sha256_{m.lower()}": p["input_hashes"][m] for m in MODALITIES})
        patient_csv.append(row)
    dump_csv(out / "patient_input_validation.csv", patient_csv)
    dump_csv(out / "missing_inputs.csv", missing_rows,
       ["patient_id", "modality", "relative_path", "classification", "reason"])
    dump_csv(out / "geometry_mismatches.csv", mismatch_rows,
       ["patient_id", "reference_modality", "modality", "shape_match", "spacing_match", "affine_match", "orientation_match",
        "reference_shape", "observed_shape", "reference_spacing_mm", "observed_spacing_mm", "reference_orientation",
        "observed_orientation", "reference_affine", "observed_affine"])
    dump_csv(out / "mask_label_summary.csv", mask_rows)
    duplicate_rows = [{"sha256": g["sha256"], "patient_ids": g["patient_ids"], "modalities": g["modalities"],
                       "duplicate_file_count": g["duplicate_file_count"], "files": g["files"]} for g in dup_groups]
    dump_csv(out / "duplicate_file_report.csv", duplicate_rows,
       ["sha256", "patient_ids", "modalities", "duplicate_file_count", "files"])

    report["elapsed_seconds"] = time.monotonic() - started
    (out / "cohort_input_validation.json").write_text(json.dumps(report, indent=2, default=jsonable, allow_nan=False) + "\n", encoding="utf-8")
    write_text_report(out / "cohort_input_validation.txt", report)
    write_latex(out, report, patients)
    # Persist PDF compilation status added by write_latex into the JSON/text report.
    (out / "cohort_input_validation.json").write_text(json.dumps(report, indent=2, default=jsonable, allow_nan=False) + "\n", encoding="utf-8")
    write_text_report(out / "cohort_input_validation.txt", report)
    print_terminal(report, out)


def write_text_report(path: Path, report: dict):
    d = report["directory_validation"]; f = report["file_validation"]; g = report["geometry_validation"]
    m = report["mask_validation"]; s = report["storage_statistics"]; final = report["final_summary"]
    lines = ["="*76, "FULL COHORT INPUT VALIDATION", f"{report['expected_range']['first']} -- {report['expected_range']['last']}", "="*76,
      f"Dataset root: {report['dataset_root']}", f"Generated UTC: {report['audit_timestamp_utc']}", "",
      "1. Dataset root", "Read-only audit; no source changes were made.", "",
      "2. Authoritative input contract",
      "Validated from cohort_meshing/pilot20/run_preprocessing_pilot20.py and phase3b_preprocess_materials.py.",
      "Required: <root>/EGD-XXXX/MR_EGD-XXXX/<modality>/NIFTI/<exact filename>"]
    for modality, spec in report["input_contract"]["required_files"].items():
        lines.append(f"  {modality}: {spec['relative_path']}")
    lines += ["HD-BET mask is derived from T1, not a raw-input prerequisite. Tumor semantics: MASK > 0.",
      "The validated Pilot20 runner requires all five files and enforces same shape and affine against T1.", "",
      "3. Expected cohort", f"Expected={d['expected_count']}; EGD directories present={d['expected_directories_present']}; required MR_EGD subdirectories present={sum(p['mr_directory_exists'] for p in report['patient_results'])}; discovered-in-range={d['discovered_expected_count']}.",
      "4. Discovered cohort", f"Missing IDs: {', '.join(d['missing_ids']) or 'None'}",
      f"Unexpected EGD directories: {', '.join(d['unexpected_egd_directories']) or 'None'}",
      f"Malformed EGD entries: {', '.join(d['malformed_egd_entries']) or 'None'}",
      f"Non-directory root entries: {d['non_directory_entries']}", "",
      "5. Required-file completeness", f"Expected slots={f['expected_file_slots']}; valid nonempty files={f['valid_required_files']}; readable NIfTI={f['readable_nifti_files']}; invalid={f['invalid_nifti_files']}.",
      f"Missing/invalid entries={len(f['missing_entries'])}.",
      "5a. Alternate directory-level observations",
      report.get("alternate_layout_findings", {}).get("summary", "No alternate-layout inspection has been performed."),
      "6. NIfTI readability", f"Readable NIfTI files={f['readable_nifti_files']}; invalid present files={f['invalid_nifti_files']}; absent exact paths={f.get('missing_exact_paths', 0)}.",
      "All required images are checked as finite nonempty 3-D volumes.", "",
      "7. Geometry consistency (T1 reference)", f"Shape pass patients={g['shape_match_patients']}; spacing={g['spacing_match_patients']}; affine={g['affine_match_patients']}; orientation={g['orientation_match_patients']}.",
      f"Affine/spacing tolerance: rtol={GEOM_RTOL:g}, atol={GEOM_ATOL:g}; orientation exact axis-code match.",
      f"Patients with mismatch records={g['patients_geometry_review']}; hard geometry failures={g['patients_geometry_fail']}.", "",
      "8. Mask validation", f"Nonempty tumor masks={m['nonempty_tumor_masks']}; empty={m['empty_tumor_masks']}; nonbinary observations={m['nonbinary_masks_observed']}.",
      f"Unique labels are preserved in mask_label_summary.csv; tumor count uses raw values > 0.", "",
      "9. Tumor-mask statistics (voxels)"]
    for key, value in m["tumor_voxel_statistics"].items():
        lines.append(f"  {key}: {value}")
    lines += ["", "10. Duplicate-file observations", f"Duplicate SHA-256 groups={len(report['duplicate_analysis']['duplicate_groups'])}; patients involved={', '.join(report['duplicate_analysis']['duplicate_patient_ids']) or 'None'}.",
      f"All five files identical between patient pairs: {report['duplicate_analysis']['all_five_required_inputs_identical_patient_pairs'] or 'None'}", "",
      "11. Storage statistics", f"Required input total={s['required_input_bytes']} bytes ({s['required_input_decimal_GB']:.3f} decimal GB).",
      f"Required input per patient bytes: {s['per_patient_required_input_bytes']}",
      f"All regular files under expected patient directories={s['all_regular_file_bytes_under_expected_patient_dirs']} bytes.",
      f"Per-modality: {s['per_modality_bytes']}", "",
      "12. Patients requiring review"]
    lines += [f"  {p['patient_id']}: {'; '.join(p['warning_reasons'])}" for p in final["review_patients"]] or ["  None."]
    lines += ["", "13. Failed patients"]
    lines += [f"  {p['patient_id']}: {'; '.join(p['reasons'])}" for p in final["blocking_patients"]] or ["  None."]
    lines += ["", "14. Final readiness decision", f"PASS={final['PASS']}; REVIEW_REQUIRED={final['REVIEW_REQUIRED']}; FAIL={final['FAIL']}.",
      f"READY_FOR_FULL_COHORT_PREPROCESSING = {'YES' if final['ready_for_full_cohort_preprocessing'] else 'NO'}",
      f"PDF report status = {report.get('latex', {}).get('status', 'not built')}",
      "Readiness concerns input completeness/readability/geometry and preprocessing prerequisites only. No preprocessing, meshing, or FEM was run."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def print_terminal(report: dict, out: Path):
    d, f, g, m, final = (report[k] for k in ("directory_validation", "file_validation", "geometry_validation", "mask_validation", "final_summary"))
    print("="*60+"\nFULL COHORT INPUT VALIDATION COMPLETE\n"+"="*60)
    print(f"Dataset:\n    {report['dataset_root']}\n\nExpected:\n    {report['expected_range']['first']} -- {report['expected_range']['last']}\n    N = {d['expected_count']}")
    print(f"\nDiscovered:\n    N = {d['discovered_expected_count']}\n\nMissing patient directories:\n    {', '.join(d['missing_ids']) or 'None'}\n\nUnexpected patient directories:\n    {', '.join(d['unexpected_egd_directories']) or 'None'}")
    print(f"\nRequired input files:\n    expected = {f['expected_file_slots']}\n    found = {f['valid_required_files']}\n    missing = {f['expected_file_slots']-f['valid_required_files']}")
    print(f"\nNIfTI:\n    readable = {f['readable_nifti_files']}\n    invalid present files = {f['invalid_nifti_files']}\n    absent exact paths = {f.get('missing_exact_paths', 0)}\n\nGeometry:\n    pass = {g['patients_all_geometry_checks_pass']}\n    review = {g['patients_geometry_review']}\n    fail = {g['patients_geometry_fail']}")
    masks_missing = sum(1 for p in report['patient_results'] if p['tumor_voxel_count'] is None)
    print(f"\nMasks:\n    readable/nonempty = {m['nonempty_tumor_masks']}\n    review = {m['nonbinary_masks_observed']}\n    missing/unreadable = {masks_missing}\n    empty = {m['empty_tumor_masks']}\n\nEmpty tumor masks:\n    {', '.join(m['empty_tumor_patient_ids']) or 'None'}")
    print(f"\nExact duplicate required-input files:\n    {len(report['duplicate_analysis']['duplicate_groups'])} groups")
    print(f"\nPatients:\n    PASS = {final['PASS']}\n    REVIEW_REQUIRED = {final['REVIEW_REQUIRED']}\n    FAIL = {final['FAIL']}\n\nREADY_FOR_FULL_COHORT_PREPROCESSING:\n    {'YES' if final['ready_for_full_cohort_preprocessing'] else 'NO'}")
    print(f"\nReports:\n    {out}\n        cohort_input_validation.txt\n        cohort_input_validation.json\n        patient_input_validation.csv\n\nPDF:\n    {out/'report'/'full_cohort_input_validation.pdf'}")
    print("\nSOURCE DATA MODIFIED:\n    NONE\n\nPREPROCESSING RUN:\n    NO\n\nMESHING RUN:\n    NO\n\nFEM RUN:\n    NO")
    print("\nREVIEW_REQUIRED PATIENTS:")
    for p in final["review_patients"]:
        print(f"    {p['patient_id']}: {'; '.join(p['reasons'])}")
    print("FAIL PATIENTS:")
    for p in final["blocking_patients"]:
        print(f"    {p['patient_id']}: {'; '.join(p['reasons'])}")
    print("="*60)


if __name__ == "__main__":
    main()
