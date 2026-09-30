#!/usr/bin/env python3
"""Resumable Full774 production orchestrator.

Dry-run is the default-safe first operation.  Production workers reuse the
validated Pilot20 preprocessing, Phase-4B mesh-construction, and independent
certification functions; this file only supplies cohort-scale paths/state.
No FEM operations are present.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/media/mazder/Games/MRI-DATA-BRAIN-774")
DEFAULT_OUTPUT = Path("/media/mazder/Games/MRI-DATA-BRAIN-774-OUTPUT")
DEFAULT_PROJECT_ROOT = Path("/home/mazder/mirp-solver/Brain-Model")
REQUIRED = {
    "T1": "1_T1/NIFTI/T1.nii.gz",
    "T1GD": "2_T1GD/NIFTI/T1GD.nii.gz",
    "T2": "3_T2/NIFTI/T2.nii.gz",
    "FLAIR": "4_FLAIR/NIFTI/FLAIR.nii.gz",
    "MASK": "5_MASK/NIFTI/MASK.nii.gz",
}
CHECKPOINTS = {20, 50, 100, 250, 500, 774}
MIN_FREE_BYTES = 50 * 10**9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    tmp.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_source(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()
    if candidate.is_dir() and all((candidate / f"EGD-{i:04d}").is_dir() for i in range(1, 775)):
        return candidate
    parent = candidate.parent if candidate.name else candidate
    options = []
    if parent.is_dir():
        for p in parent.iterdir():
            if not p.is_dir() or p == DEFAULT_OUTPUT:
                continue
            count = sum((p / f"EGD-{i:04d}").is_dir() for i in range(1, 775))
            if count == 774:
                options.append(p.resolve())
    if len(options) == 1:
        return options[0]
    raise RuntimeError(f"Could not identify one source root with EGD-0001..EGD-0774: {candidate}")


def expected_ids() -> list[str]:
    return [f"EGD-{i:04d}" for i in range(1, 775)]


def validate_source(source: Path) -> tuple[list[dict], dict]:
    rows = []
    failures = []
    for pid in expected_ids():
        patient = source / pid / f"MR_{pid}"
        row = {"patient_id": pid, "patient_dir": str(patient), "files": {}, "ready": True, "reasons": []}
        for modality, rel in REQUIRED.items():
            path = patient / rel
            record = {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": None}
            if not path.is_file() or path.stat().st_size == 0:
                row["ready"] = False
                row["reasons"].append(f"{modality}_missing_or_empty")
            else:
                record["sha256"] = sha256(path)
            row["files"][modality] = record
        if not row["ready"]:
            failures.append(row)
        rows.append(row)
    return rows, {"expected": 774, "discovered_expected": sum(r["patient_dir"] and (source / r["patient_id"]).is_dir() for r in rows), "input_ready": sum(r["ready"] for r in rows), "failures": failures}


def write_preprocessing_audit(output: Path, rows: list[dict]) -> Path:
    path = output / "manifest" / "preprocessing_input_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "T1_SHA256", "T1GD_SHA256", "T2_SHA256", "FLAIR_SHA256", "MASK_SHA256"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"patient_id": row["patient_id"], **{f"{m}_SHA256": row["files"][m]["sha256"] or "" for m in REQUIRED}})
    return path


def software_versions() -> dict:
    versions = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node()}
    for package in ("numpy", "nibabel", "scipy", "meshio"):
        try:
            mod = __import__(package)
            versions[package] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[package] = f"UNAVAILABLE: {exc}"
    return versions


def copy_paper(output: Path) -> dict:
    source = ROOT / "cohort_meshing/full774/paper"
    destination = output / "paper"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.pdf"))
        else:
            shutil.copy2(item, target)
        copied.append(str(target))
    return {"source": str(source), "destination": str(destination), "files_copied": copied}


def compile_paper(output: Path) -> tuple[bool, str]:
    paper = output / "paper"
    updater = paper / "update_paper.py"
    builder = paper / "build_paper.py"
    if not updater.is_file() or not builder.is_file():
        return False, "paper updater or builder missing"
    up = subprocess.run([sys.executable, str(updater)], cwd=paper, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if up.returncode:
        return False, up.stdout[-4000:]
    build = subprocess.run([sys.executable, str(builder)], cwd=paper, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return build.returncode == 0 and (paper / "full774_meshing_study.pdf").is_file(), build.stdout[-4000:]


def preflight(args) -> dict:
    source = discover_source(args.source_root)
    output = args.output_root.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise RuntimeError("SOURCE_ROOT and OUTPUT_ROOT are not isolated")
    rows, counts = validate_source(source)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "patients", "checkpoints", "statistics/figures", "statistics/tables", "reports", "logs", "state", "scripts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    validation_dir = output / "manifest" / "raw_input_validation"
    validation_cmd = [sys.executable, str(ROOT / "cohort_meshing/validate_full_cohort_inputs.py"), "--data-root", str(source), "--first-id", "1", "--last-id", "774", "--output-dir", str(validation_dir)]
    if args.reuse_input_validation and (validation_dir / "cohort_input_validation.json").is_file():
        validation_proc = subprocess.CompletedProcess(validation_cmd, 0, "Reused existing completed input-validation artifact")
    else:
        validation_proc = subprocess.run(validation_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    validation_json = validation_dir / "cohort_input_validation.json"
    if not validation_json.is_file():
        raise RuntimeError("authoritative full-cohort input validator did not produce JSON output\n" + validation_proc.stdout[-4000:])
    validation_report = json.loads(validation_json.read_text(encoding="utf-8"))
    validation_ready = bool(validation_report.get("final_summary", {}).get("ready_for_full_cohort_preprocessing", False)) and validation_proc.returncode == 0
    available = shutil.disk_usage(output)
    pilot20_gb = 6.278
    estimated_mesh_gb = pilot20_gb / 20.0 * 774.0
    estimated_required_gb = estimated_mesh_gb * 1.35
    code_files = [ROOT / "cohort_meshing/pilot20/run_preprocessing_pilot20.py", ROOT / "phase3b_preprocess_materials.py", ROOT / "cohort_meshing/pilot20/run_mesh_pilot20.py", ROOT / "phase4b_generate_mesh.py", ROOT / "mesh_validation/certify_authoritative_mesh.py", ROOT / "cohort_meshing/full774/paper/update_paper.py", ROOT / "cohort_meshing/full774/paper/build_paper.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in code_files if p.is_file()}
    audit_path = write_preprocessing_audit(output, rows)
    source_manifest = {"generated_utc": now(), "source_root": str(source), "expected_ids": expected_ids(), "patient_inputs": rows, "input_contract": REQUIRED}
    json_dump(output / "manifest/source_manifest.json", source_manifest)
    paper_copy = copy_paper(output)
    for script in (ROOT / "cohort_meshing/full774/run_full774.py", ROOT / "cohort_meshing/validate_full_cohort_inputs.py", *code_files):
        if script.is_file():
            shutil.copy2(script, output / "scripts" / script.name)
    paper_ok, paper_log = compile_paper(output)
    json_dump(output / "manifest/pipeline_manifest.json", {
        "cohort": "EGD-0001..EGD-0774", "expected_N": 774, "source_root": str(source), "output_root": str(output),
        "preprocessing_implementation": str(ROOT / "cohort_meshing/pilot20/run_preprocessing_pilot20.py"),
        "meshing_implementation": str(ROOT / "phase4b_generate_mesh.py"),
        "certification_implementation": str(ROOT / "mesh_validation/certify_authoritative_mesh.py"),
        "source_code_sha256": code_hashes, "software": software_versions(), "run_start_utc": now(),
        "configuration": {"workers": args.workers, "dry_run": args.dry_run, "fem": False, "checkpoints": sorted(CHECKPOINTS), "input_validation_command": validation_cmd},
        "paper_copy": paper_copy, "paper_compilation": {"pass": paper_ok, "log_tail": paper_log},
        "estimated_mesh_storage_decimal_GB": estimated_mesh_gb, "estimated_required_decimal_GB": estimated_required_gb,
    })
    state = {"phase": "FULL774_PRODUCTION", "status": "PREFLIGHT_COMPLETE", "source_root": str(source), "output_root": str(output), "expected_patients": expected_ids(), "patients": {}, "updated_utc": now()}
    json_dump(output / "state/run_state.json", state)
    report = {"source_root": str(source), "output_root": str(output), "expected_patients": 774, "discovered_patients": counts["discovered_expected"], "input_ready": int(validation_report.get("final_summary", {}).get("PASS", 0)), "input_failures": validation_report.get("final_summary", {}).get("blocking_patients", []), "input_validation_returncode": validation_proc.returncode, "input_validation_ready": validation_ready, "input_validation_report": str(validation_json), "available_bytes": available.free, "capacity_bytes": available.total, "estimated_mesh_storage_bytes": int(estimated_mesh_gb * 1e9), "estimated_required_bytes": int(estimated_required_gb * 1e9), "disk_safety": available.free > estimated_required_gb * 1e9, "pipeline_located": all(p.is_file() for p in code_files[:5]), "paper_template_located": (ROOT / "cohort_meshing/full774/paper/full774_meshing_study.tex").is_file(), "paper_compilation_pass": paper_ok, "resume_support": True, "source_output_isolated": source != output and source not in output.parents and output not in source.parents, "preprocessing_run": False, "meshing_run": False, "fem_run": False, "code_hashes": code_hashes, "generated_utc": now()}
    json_dump(output / "reports/preflight.json", report)
    return report


def print_preflight(report: dict) -> None:
    ready = report["discovered_patients"] == 774 and report["input_ready"] == 774 and report["input_validation_ready"] and report["disk_safety"] and report["pipeline_located"] and report["paper_template_located"] and report["paper_compilation_pass"] and report["resume_support"] and report["source_output_isolated"]
    print("=" * 60)
    print("FULL774 PRODUCTION PREFLIGHT COMPLETE")
    print("=" * 60)
    print(f"SOURCE_ROOT:\n    {report['source_root']}\n\nOUTPUT_ROOT:\n    {report['output_root']}\n\nExpected patients:\n    {report['expected_patients']}\n\nInput-ready:\n    {report['input_ready']}\n\nInput failures:\n    {len(report['input_failures'])}")
    print(f"\nAvailable disk:\n    {report['available_bytes'] / 1e9:.3f} GB decimal\n\nEstimated mesh storage:\n    {report['estimated_mesh_storage_bytes'] / 1e9:.3f} GB decimal\n\nSafety assessment:\n    {'PASS' if report['disk_safety'] else 'FAIL'}")
    print(f"\nPreprocessing implementation:\n    {report['code_hashes'].get('cohort_meshing/pilot20/run_preprocessing_pilot20.py', 'MISSING')}\n\nMeshing implementation:\n    {report['code_hashes'].get('phase4b_generate_mesh.py', 'MISSING')}\n\nCertification implementation:\n    {report['code_hashes'].get('mesh_validation/certify_authoritative_mesh.py', 'MISSING')}")
    print(f"\nPaper copied:\n    YES\n\nPaper compilation:\n    {'PASS' if report['paper_compilation_pass'] else 'FAIL'}\n\nResume support:\n    {'PASS' if report['resume_support'] else 'FAIL'}\n\nSource/output isolation:\n    {'PASS' if report['source_output_isolated'] else 'FAIL'}")
    print("\nPatient data modified:\n    NONE\n\nPreprocessing run:\n    NO\n\nMeshing run:\n    NO\n\nFEM run:\n    NO")
    print(f"\nREADY_FOR_FULL774_PRODUCTION:\n    {'YES' if ready else 'NO'}")
    if not ready:
        blockers = []
        if report["discovered_patients"] != 774: blockers.append("expected patient directory set is incomplete")
        if report["input_ready"] != 774 or not report["input_validation_ready"]: blockers.append("mandatory raw input readability/geometry failure(s)")
        if not report["disk_safety"]: blockers.append("insufficient disk safety margin")
        if not report["pipeline_located"]: blockers.append("validated pipeline component missing")
        if not report["paper_compilation_pass"]: blockers.append("production paper compilation failed")
        print("\nBlockers:")
        for item in blockers: print(f"    {item}")
    print("=" * 60)


def run_production(args):
    source = discover_source(args.source_root)
    output = args.output_root.expanduser().resolve()
    manifest = json.loads((output / "manifest/source_manifest.json").read_text(encoding="utf-8"))
    audit_path = output / "manifest/preprocessing_input_audit.csv"
    if not audit_path.is_file():
        raise RuntimeError("Preflight manifest missing; run --dry-run first")
    pre = load_module(ROOT / "cohort_meshing/pilot20/run_preprocessing_pilot20.py", "full774_preprocessing")
    mesh = load_module(ROOT / "cohort_meshing/pilot20/run_mesh_pilot20.py", "full774_mesh_helpers")
    mesh_gen = load_module(ROOT / "phase4b_generate_mesh.py", "full774_mesh_generator")
    pre.RAW_ROOT = source
    pre.AUDIT = audit_path
    pre.OUT_ROOT = output / "patients"
    pre.PATIENT_OUTPUT_SUBDIR = True
    audit_rows = pre.load_audit()
    state_path = output / "state/run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"patients": {}}
    state.setdefault("patients", {})
    start = time.monotonic()
    consecutive_failures = []
    for index, pid in enumerate(expected_ids(), 1):
        pdir = output / "patients" / pid
        summary_path = pdir / "patient_summary.json"
        prior = state["patients"].get(pid, {})
        if args.resume and prior.get("terminal") and prior.get("status") in {"CERTIFIED", "CERTIFIED_WITH_OBSERVATION"} and summary_path.is_file():
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            pre_report = pre.patient_preprocess(pid, audit_rows[pid])
            pre_status = pre_report.get("status")
            pre_failures = set(pre_report.get("failure_invariants", pre_report.get("failures", [])))
            known_topology_observation = pre_failures == {"no_internal_background_cavities"} and int(pre_report.get("internal_background", {}).get("voxels", 0)) > 0
            if pre_status != "PREPROCESSING_CERTIFIED" and not known_topology_observation:
                summary = {"patient_id": pid, "preprocessing_status": pre_status, "final_status": "PREPROCESS_FAIL", "failure_stage": pre_report.get("stage"), "failure_reason": pre_report.get("failure_invariants", pre_report.get("failures", [])), "timings": {"patient_end_to_end_seconds": time.monotonic() - started}, "FEM": "NOT_RUN"}
                json_dump(summary_path, summary)
                state["patients"][pid] = {"terminal": True, "status": "PREPROCESS_FAIL", "updated_utc": now()}
                json_dump(state_path, state)
                continue
            label_path = pdir / "preprocessing/material_labels.nii.gz"
            label_img = __import__("nibabel").load(str(label_path))
            labels = __import__("numpy").asanyarray(label_img.dataobj)
            affine = __import__("numpy").asarray(label_img.affine, dtype=float)
            mesh_dir = pdir / "mesh"; cert_dir = pdir / "certification"; log_dir = pdir / "logs"
            mesh_dir.mkdir(exist_ok=True); cert_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
            mesh_path = mesh_dir / "head_tumor_5region.msh"
            if not (args.resume and mesh_path.is_file()):
                coords, old_ids, mapping = mesh_gen.compact_mapping(labels)
                points = mesh_gen.corner_points(old_ids, labels.shape, affine)
                positive = float(__import__("numpy").linalg.det(affine[:3, :3])) > 0
                hexes = mapping[mesh_gen.hex_conn(coords, labels.shape, positive)]
                hex_tags = labels[tuple(coords.T)].astype(__import__("numpy").int32)
                qold, qtag = mesh_gen.interface_arrays(labels, labels.shape)
                quads = mapping[qold]
                if (quads < 0).any(): raise RuntimeError("interface references unreferenced node")
                part = mesh_path.with_suffix(mesh_path.suffix + ".part")
                mesh_gen.write_msh(points, hexes, hex_tags, quads, qtag, part)
                part.replace(mesh_path)
                del coords, old_ids, mapping, points, hexes, hex_tags, qold, qtag, quads
                gc.collect()
            labels2 = __import__("nibabel").load(str(label_path))
            labels2_arr = __import__("numpy").asanyarray(labels2.dataobj)
            cert = mesh.cert_mesh(mesh_path, labels2_arr, affine, mesh.sha(mesh_path))
            expected_hex = int(__import__("numpy").count_nonzero(labels2_arr))
            gates = {
                "hex8_count": cert["hex8"] == expected_hex,
                "material_counts": cert["hex8_counts_match_labels"],
                "interface_counts": cert["interface_count_reconciliation_exact"],
                "image_reconstruction": cert["material_reconstruction_exact"],
                "tumor": cert["tumor_exact"],
                "positive_geometry": cert["inverted_hex8"] == 0 and cert["zero_jacobian_hex8"] == 0 and cert["minimum_signed_jacobian"] > 0,
                "no_duplicates": cert["duplicate_hex8"] == 0 and cert["duplicate_quad4"] == 0,
                "no_orphans": cert["orphan_mesh_nodes"] == 0,
                "no_forbidden_contacts": cert["forbidden_material_contacts"] == 0,
                "no_nonmanifold": cert["non_manifold_faces"] == 0,
                "physical_names": cert["physical_names_valid"],
                "coordinate_map": cert["all_nodes_on_expected_affine_lattice"],
            }
            passed = all(gates.values())
            observation = bool(pre_report.get("internal_background", {}).get("voxels", 0))
            observation = observation or known_topology_observation
            classification = "CERTIFIED_WITH_OBSERVATION" if passed and observation else "CERTIFIED" if passed else "CERTIFICATION_FAIL"
            summary = {"patient_id": pid, "preprocessing_status": "PREPROCESSING_CERTIFIED_WITH_OBSERVATION" if known_topology_observation else pre_status, "mesh_status": "MESH_GENERATED", "certification_status": classification, "final_status": classification, "source_input_hashes": pre_report.get("raw_hashes", {}), "material_labels_sha256": sha256(label_path), "mesh_path": str(mesh_path), "mesh_sha256": sha256(mesh_path), "mesh_size_bytes": mesh_path.stat().st_size, "certificate": cert, "certification_gates": gates, "observation_count": int(observation), "timings": {"preprocessing_seconds": pre_report.get("runtime_seconds"), "patient_end_to_end_seconds": time.monotonic() - started}, "peak_RSS_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024), "FEM": "NOT_RUN"}
            json_dump(cert_dir / "mesh_certificate.json", {**cert, "patient_id": pid, "classification": classification, "certification_gates": gates})
            (cert_dir / "mesh_certificate.txt").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            json_dump(summary_path, summary)
            state["patients"][pid] = {"terminal": True, "status": classification, "updated_utc": now()}
            consecutive_failures = [] if passed else consecutive_failures + [(pid, "certification")]
        except Exception as exc:
            summary = {"patient_id": pid, "final_status": "MESH_FAIL", "failure_stage": "exception", "failure_reason": repr(exc), "traceback": traceback.format_exc(), "timings": {"patient_end_to_end_seconds": time.monotonic() - started}, "FEM": "NOT_RUN"}
            json_dump(summary_path, summary)
            state["patients"][pid] = {"terminal": True, "status": "MESH_FAIL", "updated_utc": now()}
            consecutive_failures.append((pid, "exception"))
        json_dump(state_path, state)
        if len(consecutive_failures) >= 3 and consecutive_failures[-1][1] == consecutive_failures[-2][1] == consecutive_failures[-3][1]:
            raise RuntimeError("POSSIBLE_SYSTEMATIC_PIPELINE_FAILURE: three consecutive same-stage failures")
        if index in CHECKPOINTS:
            write_full_summary(output, start)
            compile_paper(output)
        if shutil.disk_usage(output).free < MIN_FREE_BYTES:
            raise RuntimeError(f"Filesystem safety stop: less than {MIN_FREE_BYTES / 1e9:.1f} GB free")
        gc.collect()
    write_full_summary(output, start)
    compile_paper(output)


def write_full_summary(output: Path, started: float) -> None:
    rows = []
    for pid in expected_ids():
        p = output / "patients" / pid / "patient_summary.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"patient_id": pid, "final_status": "NOT_PROCESSED"}
        cert = data.get("certificate", {})
        mats = cert.get("hex8_by_material", {})
        quads = cert.get("quad4_by_tag", {})
        rows.append({"patient_id": pid, "preprocessing_status": data.get("preprocessing_status"), "certification_status": data.get("certification_status", data.get("final_status")), "HEX8_total": cert.get("hex8"), "QUAD4_total": cert.get("quad4"), "total_elements": (cert.get("hex8", 0) or 0) + (cert.get("quad4", 0) or 0), "BrainTissue_HEX8": mats.get("1"), "Tumor_HEX8": mats.get("2"), "CSF_HEX8": mats.get("3"), "Skull_HEX8": mats.get("4"), "Scalp_HEX8": mats.get("5"), "tag101_faces": quads.get("101"), "tag102_faces": quads.get("102"), "tag103_faces": quads.get("103"), "tag104_faces": quads.get("104"), "tag105_faces": quads.get("105"), "material_reconstruction_differences": cert.get("image_reconstruction_differing_voxels"), "tumor_reconstruction_differences": cert.get("tumor_differing_voxels"), "nonmanifold_faces": cert.get("non_manifold_faces"), "forbidden_contacts": cert.get("forbidden_material_contacts"), "mesh_size_bytes": data.get("mesh_size_bytes"), "peak_RSS_bytes": data.get("peak_RSS_bytes"), "mesh_SHA256": data.get("mesh_sha256"), "failure_reason": data.get("failure_reason")})
    csv_path = output / "full774_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    statuses = [r["certification_status"] for r in rows]
    report = {"cohort": "EGD-0001..EGD-0774", "expected": 774, "processed": sum(s != "NOT_PROCESSED" for s in statuses), "certified": sum(s in {"CERTIFIED", "CERTIFIED_WITH_OBSERVATION"} for s in statuses), "certified_with_observation": statuses.count("CERTIFIED_WITH_OBSERVATION"), "failed": sum(s in {"PREPROCESS_FAIL", "MESH_FAIL", "CERTIFICATION_FAIL"} for s in statuses), "not_processed": statuses.count("NOT_PROCESSED"), "total_hex8": sum(r["HEX8_total"] or 0 for r in rows), "total_quad4": sum(r["QUAD4_total"] or 0 for r in rows), "total_elements": sum(r["total_elements"] or 0 for r in rows), "total_runtime_seconds": time.monotonic() - started, "FEM_RUN": "NO", "complete": all(s != "NOT_PROCESSED" for s in statuses), "patient_results": rows}
    json_dump(output / "full774_report.json", report)
    (output / "full774_report.txt").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None, help="project containing validated implementations")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="perform preflight only")
    parser.add_argument("--reuse-input-validation", action="store_true", help="reuse an existing completed validator artifact when refreshing manifests")
    parser.add_argument("--run-production", action="store_true", help="reserved for explicit later launch")
    parser.add_argument("--resume", action="store_true", help="reuse terminal patient states and continue incomplete cohort")
    args = parser.parse_args()
    if args.project_root is not None:
        ROOT = args.project_root.expanduser().resolve()
    elif not (ROOT / "cohort_meshing").is_dir():
        ROOT = DEFAULT_PROJECT_ROOT.resolve()
    if args.run_production:
        run_production(args)
        return 0
    report = preflight(args)
    print_preflight(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
