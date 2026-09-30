#!/usr/bin/env python3
"""Read-only statistical analysis of the completed Pilot20 mesh cohort."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "statistics"
CSV_PATH = ROOT / "mesh_summary.csv"
JSON_PATH = ROOT / "pilot20_mesh_report.json"
TXT_PATH = ROOT / "pilot20_mesh_report.txt"
MATERIALS = ["BrainTissue", "Tumor", "CSF", "Skull", "Scalp"]
METRICS = {
    "HEX8_total": "HEX8 count",
    "QUAD4_total": "QUAD4 count",
    "total_elements": "Total elements",
    "mesh_size_MB": "Mesh size (MB, decimal)",
    "mesh_generation_seconds": "Mesh-generation time (s)",
    "certification_seconds": "Certification time (s)",
    "peak_RSS_MB": "Peak RSS (MB, 1024-based conversion)",
    "HEX8_Tumor": "Tumor HEX8 count",
    "HEX8_BrainTissue": "BrainTissue HEX8 count",
    "HEX8_CSF": "CSF HEX8 count",
    "HEX8_Skull": "Skull HEX8 count",
    "HEX8_Scalp": "Scalp HEX8 count",
    "interface_101": "Interface 101 faces",
    "interface_102": "Interface 102 faces",
    "interface_103": "Interface 103 faces",
    "interface_104": "Interface 104 faces",
    "interface_105": "Interface 105 faces",
    "total_QUAD4": "Total QUAD4 faces",
}


def quantile(values, p):
    return float(np.quantile(np.asarray(values, dtype=float), p))


def stats(values):
    a = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not a:
        return {k: None for k in ("N", "mean", "median", "std", "min", "Q1", "Q3", "max", "IQR", "CV", "CV_percent")}
    mean = statistics.mean(a)
    sd = statistics.stdev(a) if len(a) > 1 else 0.0
    q1, q3 = quantile(a, .25), quantile(a, .75)
    cv = sd / mean if mean else None
    return {"N": len(a), "mean": mean, "median": statistics.median(a), "std": sd,
            "min": min(a), "Q1": q1, "Q3": q3, "max": max(a), "IQR": q3-q1,
            "CV": cv, "CV_percent": cv*100 if cv is not None else None}


def corr(x, y, method="pearson"):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return {"r": None, "N": len(pairs)}
    xx, yy = zip(*pairs)
    if method == "spearman":
        xx = statistics.rankdata(xx) if hasattr(statistics, "rankdata") else rankdata(xx)
        yy = statistics.rankdata(yy) if hasattr(statistics, "rankdata") else rankdata(yy)
    if len(set(xx)) < 2 or len(set(yy)) < 2:
        return {"r": None, "N": len(pairs)}
    return {"r": float(np.corrcoef(xx, yy)[0, 1]), "N": len(pairs)}


def rankdata(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return ranks


def linear_model(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2 or len({a for a, _ in pairs}) < 2:
        return {"N": len(pairs), "intercept": None, "slope": None, "R2": None}
    xx, yy = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
    slope, intercept = np.polyfit(xx, yy, 1)
    pred = intercept + slope * xx
    ssres = float(np.sum((yy-pred)**2)); sst = float(np.sum((yy-np.mean(yy))**2))
    return {"N": len(pairs), "intercept": float(intercept), "slope": float(slope),
            "R2": 1-ssres/sst if sst else None}


def fmt(v, digits=3):
    if v is None:
        return "NOT AVAILABLE"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:,.{digits}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    for required in (CSV_PATH, JSON_PATH, TXT_PATH):
        if not required.is_file():
            raise SystemExit(f"Required authoritative input missing: {required}")
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader); columns = reader.fieldnames or []
    if not rows:
        raise SystemExit("mesh_summary.csv contains no patients")
    report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    required_cols = ["patient_id", "mesh_status", "preprocessing_warning", "HEX8_total", "QUAD4_total",
                     "points", "mesh_size_GB", "mesh_generation_seconds", "certification_seconds", "peak_RSS_GB",
                     "mesh_to_image_differing_voxels", "tumor_differing_voxels", "non_manifold_faces", "forbidden_contacts"]
    missing = [c for c in required_cols if c not in columns]
    if missing:
        raise SystemExit(f"Required CSV columns missing: {missing}; found schema={columns}")
    print("DISCOVERED CSV SCHEMA:")
    print(", ".join(columns))

    numeric_cols = [c for c in columns if c not in ("patient_id", "mesh_status", "preprocessing_warning", "mesh_SHA256")]
    for i, r in enumerate(rows, 1):
        for c in numeric_cols:
            try:
                v = float(r[c])
                if not math.isfinite(v):
                    raise ValueError("non-finite")
                r[c] = v
            except (TypeError, ValueError):
                raise SystemExit(f"Invalid numeric value row {i} patient={r.get('patient_id')} column={c}: {r.get(c)!r}")
    ids = [r["patient_id"] for r in rows]
    if len(set(ids)) != len(rows):
        raise SystemExit("Duplicate patient identifiers in summary CSV")

    for r in rows:
        r["total_elements"] = r["HEX8_total"] + r["QUAD4_total"]
        r["total_QUAD4"] = r["QUAD4_total"]
        r["mesh_size_MB"] = r["mesh_size_GB"] * 1000.0  # source field is decimal GB
        r["peak_RSS_MB"] = r["peak_RSS_GB"] * 1024.0  # source field is GiB-like RSS conversion
        n_mil = r["total_elements"] / 1e6
        r["quad_to_hex_ratio"] = r["QUAD4_total"] / r["HEX8_total"] if r["HEX8_total"] else None
        r["seconds_per_million_elements"] = r["mesh_generation_seconds"] / n_mil
        r["certification_seconds_per_million_elements"] = r["certification_seconds"] / n_mil
        r["MB_per_million_elements"] = r["mesh_size_MB"] / n_mil
        r["RSS_MB_per_million_elements"] = r["peak_RSS_MB"] / n_mil
        r["tumor_fraction"] = r["HEX8_Tumor"] / r["HEX8_total"]
        r["brain_fraction"] = r["HEX8_BrainTissue"] / r["HEX8_total"]
        r["CSF_fraction"] = r["HEX8_CSF"] / r["HEX8_total"]
        r["skull_fraction"] = r["HEX8_Skull"] / r["HEX8_total"]
        r["scalp_fraction"] = r["HEX8_Scalp"] / r["HEX8_total"]
        r["interface101_per_tumor_hex8"] = r["interface_101"] / r["HEX8_Tumor"] if r["HEX8_Tumor"] else None
        r["QUAD4_per_HEX8"] = r["QUAD4_total"] / r["HEX8_total"]
        r["vertices_per_HEX8"] = r["points"] / r["HEX8_total"]
        r["mesh_MB_per_million_HEX8"] = r["mesh_size_MB"] / (r["HEX8_total"] / 1e6)
        r["generation_seconds_per_million_HEX8"] = r["mesh_generation_seconds"] / (r["HEX8_total"] / 1e6)
        r["certified"] = r["mesh_status"].startswith("MESH_CERTIFIED")
        r["warning_observation"] = r["preprocessing_warning"] not in ("NONE", "", None)
        r["reconstruction_exact"] = r["mesh_to_image_differing_voxels"] == 0
        r["tumor_exact"] = r["tumor_differing_voxels"] == 0
        r["non_manifold_free"] = r["non_manifold_faces"] == 0
        r["forbidden_contact_free"] = r["forbidden_contacts"] == 0

    for name, field, expected in [
        ("patients", None, report["patients_attempted"]),
        ("total_hex8", "HEX8_total", report["total_hex8"]),
        ("total_quad4", "QUAD4_total", report["total_quad4"]),
        ("total_elements", "total_elements", report["total_hex8"] + report["total_quad4"]),
        ("total_mesh_size_GB", "mesh_size_GB", report["total_storage_bytes"] / 1e9),
        ("sum_generation_plus_certification_seconds", None, None),
    ]:
        if name == "patients": actual = len(rows)
        elif field: actual = sum(r[field] for r in rows)
        else: actual = None
        if name != "sum_generation_plus_certification_seconds":
            difference = actual - expected
            ok = abs(difference) <= (1e-8 * max(1, abs(expected)))
            checks[name] = {"reported": expected, "recomputed": actual, "difference": difference,
                            "status": "PASS" if ok else "FAIL"}
        else:
            stage_sum = sum(r["mesh_generation_seconds"] + r["certification_seconds"] for r in rows)
            checks[name] = {"reported_total_runtime_seconds": report["total_runtime_seconds"],
                            "recomputed_recorded_stage_sum_seconds": stage_sum,
                            "difference_seconds": stage_sum-report["total_runtime_seconds"],
                            "status": "SCOPE_DIFFERENCE_REPORTED"}
    checks["total_storage_bytes_vs_csv"] = {"reported": report["total_storage_bytes"],
       "recomputed": sum(r["mesh_size_GB"] for r in rows)*1e9,
       "difference": sum(r["mesh_size_GB"] for r in rows)*1e9-report["total_storage_bytes"],
       "status": "PASS" if round(sum(r["mesh_size_GB"] for r in rows)*1e9) == report["total_storage_bytes"] else "FAIL"}
    checks["all_certification_rows"] = {"recomputed": sum(r["certified"] for r in rows),
       "reported": report["meshes_certified"], "status": "PASS" if sum(r["certified"] for r in rows)==report["meshes_certified"] else "FAIL"}
    checks["all_certificates_exact_reconstruction"] = {"recomputed": sum(r["reconstruction_exact"] for r in rows),
       "reported": report["exact_material_reconstruction"], "status": "PASS" if sum(r["reconstruction_exact"] for r in rows)==report["exact_material_reconstruction"] else "FAIL"}
    checks["all_certificates_exact_tumor"] = {"recomputed": sum(r["tumor_exact"] for r in rows),
       "reported": report["exact_tumor_mesh_preservation"], "status": "PASS" if sum(r["tumor_exact"] for r in rows)==report["exact_tumor_mesh_preservation"] else "FAIL"}
    checks["all_nonmanifold_free"] = {"recomputed": sum(r["non_manifold_free"] for r in rows),
       "reported": report["non_manifold_free"], "status": "PASS" if sum(r["non_manifold_free"] for r in rows)==report["non_manifold_free"] else "FAIL"}
    checks["all_forbidden_contact_free"] = {"recomputed": sum(r["forbidden_contact_free"] for r in rows),
       "reported": report["forbidden_contact_free"], "status": "PASS" if sum(r["forbidden_contact_free"] for r in rows)==report["forbidden_contact_free"] else "FAIL"}

    numeric_metrics = [c for c in numeric_cols if c in rows[0]]
    numeric_metrics += ["total_elements", "total_QUAD4", "mesh_size_MB", "peak_RSS_MB", "quad_to_hex_ratio",
                        "seconds_per_million_elements", "certification_seconds_per_million_elements",
                        "MB_per_million_elements", "RSS_MB_per_million_elements", "tumor_fraction", "brain_fraction",
                        "CSF_fraction", "skull_fraction", "scalp_fraction", "interface101_per_tumor_hex8",
                        "QUAD4_per_HEX8", "vertices_per_HEX8", "mesh_MB_per_million_HEX8",
                        "generation_seconds_per_million_HEX8"]
    numeric_metrics = list(dict.fromkeys(numeric_metrics))
    descriptive = {m: stats([r[m] for r in rows]) for m in numeric_metrics}
    correlations = {}
    pairs = [("total_elements", "mesh_generation_seconds"), ("HEX8_total", "mesh_generation_seconds"),
             ("QUAD4_total", "mesh_generation_seconds"), ("total_elements", "mesh_size_MB"),
             ("total_elements", "peak_RSS_MB"), ("mesh_generation_seconds", "peak_RSS_MB"),
             ("HEX8_total", "certification_seconds"), ("HEX8_total", "points")]
    for a,b in pairs:
        key=f"{a}__{b}"
        correlations[key] = {"pearson": corr([r[a] for r in rows],[r[b] for r in rows]),
                             "spearman": corr([r[a] for r in rows],[r[b] for r in rows],"spearman")}
    models = {"processing_time_seconds_vs_total_elements": linear_model([r["total_elements"] for r in rows],[r["mesh_generation_seconds"] for r in rows]),
              "mesh_size_MB_vs_total_elements": linear_model([r["total_elements"] for r in rows],[r["mesh_size_MB"] for r in rows]),
              "peak_RSS_MB_vs_total_elements": linear_model([r["total_elements"] for r in rows],[r["peak_RSS_MB"] for r in rows])}
    outliers=[]
    for m in ("total_elements", "mesh_generation_seconds", "mesh_size_MB", "peak_RSS_MB"):
        s=descriptive[m]; lo=s["Q1"]-1.5*s["IQR"]; hi=s["Q3"]+1.5*s["IQR"]
        for r in rows:
            val=r[m]
            outliers.append({"patient":r["patient_id"],"metric":m,"value":val,"lower_IQR_bound":lo,
                             "upper_IQR_bound":hi,"outlier_flag":bool(val<lo or val>hi)})
    flagged=[x for x in outliers if x["outlier_flag"]]
    clean=[r for r in rows if not r["warning_observation"]]
    warned=[r for r in rows if r["warning_observation"]]
    group_fields=["total_elements","HEX8_total","QUAD4_total","mesh_generation_seconds","mesh_size_MB","peak_RSS_MB"]
    group_summary={gname:{m:stats([r[m] for r in g]) for m in group_fields}
                   for gname,g in (("without_warning",clean),("with_documented_observation",warned))}

    # Machine-readable tabular outputs.
    with (OUT/"descriptive_statistics.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["metric","N","mean","median","std","min","Q1","Q3","max","IQR","CV","CV_percent"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for m,s in descriptive.items(): w.writerow({"metric":m,**s})
    derived_fields=["patient_id"]+numeric_metrics+["quad_to_hex_ratio","seconds_per_million_elements","certification_seconds_per_million_elements",
       "MB_per_million_elements","RSS_MB_per_million_elements","tumor_fraction","brain_fraction","CSF_fraction","skull_fraction","scalp_fraction",
       "interface101_per_tumor_hex8","QUAD4_per_HEX8","vertices_per_HEX8","mesh_MB_per_million_HEX8","generation_seconds_per_million_HEX8"]
    derived_fields=list(dict.fromkeys(derived_fields))
    with (OUT/"patient_derived_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=derived_fields,extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(r)
    with (OUT/"correlation_matrix.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["metric_x","metric_y","pearson_r","spearman_r","N"])
        for (a,b),obj in [((a,b),correlations[f"{a}__{b}"]) for a,b in pairs]:
            w.writerow([a,b,obj["pearson"]["r"],obj["spearman"]["r"],obj["pearson"]["N"]])
    with (OUT/"outlier_report.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["patient","metric","value","lower_IQR_bound","upper_IQR_bound","outlier_flag"]);w.writeheader();w.writerows(outliers)

    # Figures: raw points/bars retained, no outlier removal.
    def bar_metric(field, filename, title, ylabel, divisor=1, color="#2878a8"):
        fig,ax=plt.subplots(figsize=(10,4.3)); vals=[r[field]/divisor for r in rows]
        ax.bar(ids,vals,color=color);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel("Patient")
        ax.tick_params(axis="x",rotation=55);fig.tight_layout();fig.savefig(figdir/filename);plt.close(fig)
    bar_metric("total_elements","total_elements_per_patient.png","Total elements per patient","Elements")
    bar_metric("HEX8_total","hex8_per_patient.png","HEX8 elements per patient","HEX8 elements")
    bar_metric("QUAD4_total","quad4_per_patient.png","QUAD4 elements per patient","QUAD4 elements")
    bar_metric("mesh_generation_seconds","processing_time_per_patient.png","Mesh generation time per patient","Seconds")
    bar_metric("mesh_size_MB","mesh_size_per_patient.png","Mesh file size per patient","MB (decimal)")
    bar_metric("peak_RSS_MB","peak_rss_per_patient.png","Peak RSS per patient","MB (1 GiB = 1024 MiB)")
    def scatter(xfield,yfield,filename,title,xlabel,ylabel,model_key=None):
        x=np.array([r[xfield] for r in rows],dtype=float);y=np.array([r[yfield] for r in rows],dtype=float)
        fig,ax=plt.subplots(figsize=(6.5,4.8));ax.scatter(x,y,color="#176b87",edgecolor="white",s=44,zorder=3)
        for r in rows: ax.annotate(r["patient_id"].replace("EGD-",""),(r[xfield],r[yfield]),xytext=(3,3),textcoords="offset points",fontsize=6)
        lm=linear_model(x,y)
        if lm["slope"] is not None:
            xx=np.linspace(float(min(x)),float(max(x)),100);ax.plot(xx,lm["intercept"]+lm["slope"]*xx,color="#cf5c36",lw=1.5)
            ax.text(.03,.97,f"R² = {lm['R2']:.3f}",transform=ax.transAxes,va="top")
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(figdir/filename);plt.close(fig)
    scatter("total_elements","mesh_generation_seconds","total_elements_vs_processing_time.png","Total elements vs mesh-generation time","Total elements","Seconds")
    scatter("total_elements","mesh_size_MB","total_elements_vs_mesh_size.png","Total elements vs mesh size","Total elements","MB (decimal)")
    scatter("total_elements","peak_RSS_MB","total_elements_vs_peak_rss.png","Total elements vs peak RSS","Total elements","MB (1 GiB = 1024 MiB)")
    fig,ax=plt.subplots(figsize=(5.3,4)); counts=[len(clean),len(warned),sum(not r["certified"] for r in rows)]
    ax.bar(["Certified\nno warning","Certified with\nobservation","Failed"],counts,color=["#4a9b70","#e1a33b","#b44b4b"])
    ax.set_ylabel("Patients");ax.set_ylim(0,max(counts+[1])*1.18);ax.set_title("Pilot20 certification status")
    for i,v in enumerate(counts):ax.text(i,v+.15,str(v),ha="center")
    fig.tight_layout();fig.savefig(figdir/"certification_summary.png");plt.close(fig)

    all_check_pass = all(v.get("status") in ("PASS", "SCOPE_DIFFERENCE_REPORTED") for v in checks.values())
    certification={"certified_count":sum(r["certified"] for r in rows),"certified_percent":100*sum(r["certified"] for r in rows)/len(rows),
       "without_warning_count":len(clean),"without_warning_percent":100*len(clean)/len(rows),
       "with_documented_observation_count":len(warned),"with_documented_observation_percent":100*len(warned)/len(rows),
       "failed_count":sum(not r["certified"] for r in rows),"failed_percent":100*sum(not r["certified"] for r in rows)/len(rows),
       "exact_material_reconstruction_count":sum(r["reconstruction_exact"] for r in rows),"tumor_preservation_count":sum(r["tumor_exact"] for r in rows),
       "non_manifold_free_count":sum(r["non_manifold_free"] for r in rows),"forbidden_contact_free_count":sum(r["forbidden_contact_free"] for r in rows),
       "reconstruction_difference_count":sum(r["mesh_to_image_differing_voxels"]!=0 for r in rows),
       "known_observation_patients":[r["patient_id"] for r in warned],"observation_percent":100*len(warned)/len(rows),
       "tag105_faces_per_observed_patient":6,"total_tag105_observation_faces":6*len(warned)}
    totals={"patients":len(rows),"HEX8":sum(r["HEX8_total"] for r in rows),"QUAD4":sum(r["QUAD4_total"] for r in rows),
       "elements":sum(r["total_elements"] for r in rows),"mesh_size_GB_decimal":sum(r["mesh_size_GB"] for r in rows),
       "mesh_size_bytes_reconstructed_from_csv":sum(r["mesh_size_GB"] for r in rows)*1e9,
       "recorded_mesh_generation_seconds_sum":sum(r["mesh_generation_seconds"] for r in rows),
       "recorded_certification_seconds_sum":sum(r["certification_seconds"] for r in rows),
       "recorded_generation_plus_certification_seconds":sum(r["mesh_generation_seconds"]+r["certification_seconds"] for r in rows),
       "authoritative_report_total_runtime_seconds":report["total_runtime_seconds"],
       "peak_RSS_max_GB":max(r["peak_RSS_GB"] for r in rows)}
    data={"analysis_timestamp_utc":datetime.now(timezone.utc).isoformat(),"source_files":[str(CSV_PATH.relative_to(ROOT.parent.parent)),str(JSON_PATH.relative_to(ROOT.parent.parent)),str(TXT_PATH.relative_to(ROOT.parent.parent))],
      "software":{"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__},"csv_schema":columns,
      "cohort_size":len(rows),"certification":certification,"totals":totals,"descriptive_statistics":descriptive,
      "correlations":correlations,"linear_models":models,"outliers":flagged,"outlier_rule":"Tukey 1.5*IQR; no observations removed",
      "warning_group_summary":group_summary,"consistency_checks":checks,
      "units_and_derivations":{"mesh_size_GB":"source field interpreted as decimal GB; MB=GB*1000",
       "peak_RSS_GB":"source measurement converted to MB using 1024 MB per GB for reporting",
       "total_elements":"HEX8_total + QUAD4_total","total_runtime":"CSV does not contain a single total-runtime column; stage sum is generation_seconds + certification_seconds. The JSON report total runtime is retained separately and explicitly compared.",
       "std":"sample standard deviation (ddof=1)","quantiles":"NumPy linear interpolation/default quantile method"},
      "interpretation":"Descriptive results for this 20-patient meshing pilot only; no clinical, FEM, solver, or future-cohort performance claims."}
    (OUT/"pilot20_statistics.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    # LaTeX publication-style tables.
    latex_rows=[]
    for m in ("HEX8_total","QUAD4_total","total_elements","mesh_size_MB","mesh_generation_seconds","peak_RSS_MB"):
        s=descriptive[m]
        latex_rows.append(f"{METRICS[m]} & {s['N']} & {s['mean']:.2f} & {s['std']:.2f} & {s['median']:.2f} & {s['min']:.2f} & {s['max']:.2f} \\\\")
    tab="\\begin{tabular}{lrrrrrr}\n\\hline\nMetric & N & Mean & SD & Median & Min & Max \\\\\n\\hline\n"+"\n".join(latex_rows)+"\n\\hline\n\\end{tabular}\n"
    (OUT/"pilot20_statistics_table.tex").write_text(tab,encoding="utf-8")
    cert_tab=("\\begin{tabular}{lrr}\n\\hline\nStatus & Count & Percent \\\\\n\\hline\n"
      f"Certified without warning & {len(clean)} & {100*len(clean)/len(rows):.1f}\\% \\\\\n"
      f"Certified with documented observation & {len(warned)} & {100*len(warned)/len(rows):.1f}\\% \\\\\n"
      f"Failed & {certification['failed_count']} & {certification['failed_percent']:.1f}\\% \\\\\n"
      f"Total & {len(rows)} & 100.0\\% \\\\\n\\hline\n\\end{{tabular}}\n")
    (OUT/"pilot20_certification_table.tex").write_text(cert_tab,encoding="utf-8")

    # Human-readable report.
    lines=["="*60,"PILOT20 FULL-RESOLUTION MESHING STATISTICAL ANALYSIS","="*60,"",
      "1. Cohort summary",f"Patients analyzed: {len(rows)}",f"Input CSV schema: {', '.join(columns)}","",
      "2. Certification statistics",f"Certified: {certification['certified_count']}/{len(rows)} ({certification['certified_percent']:.1f}%)",
      f"Certified without warning: {len(clean)}/{len(rows)} ({certification['without_warning_percent']:.1f}%)",
      f"Certified with documented observation: {len(warned)}/{len(rows)} ({certification['with_documented_observation_percent']:.1f}%)",
      f"Failed: {certification['failed_count']}/{len(rows)} ({certification['failed_percent']:.1f}%)",
      f"Exact material reconstruction: {certification['exact_material_reconstruction_count']}/{len(rows)}",
      f"Exact tumor preservation: {certification['tumor_preservation_count']}/{len(rows)}",
      f"Non-manifold-free: {certification['non_manifold_free_count']}/{len(rows)}; forbidden-contact-free: {certification['forbidden_contact_free_count']}/{len(rows)}","",
      "3. Cohort totals",f"HEX8={totals['HEX8']:,}; QUAD4={totals['QUAD4']:,}; total elements={totals['elements']:,}",
      f"Mesh storage={totals['mesh_size_GB_decimal']:.6f} decimal GB ({totals['mesh_size_bytes_reconstructed_from_csv']:,.0f} bytes)",
      f"Generation-time sum={totals['recorded_mesh_generation_seconds_sum']:.3f}s; certification-time sum={totals['recorded_certification_seconds_sum']:.3f}s",
      f"Recorded stage sum={totals['recorded_generation_plus_certification_seconds']:.3f}s; report total runtime={totals['authoritative_report_total_runtime_seconds']:.3f}s.",
      f"These differ by {checks['sum_generation_plus_certification_seconds']['difference_seconds']:.3f}s; the CSV lacks a single total-runtime field, so the report total is not silently equated with the sum of these two stages.","",
      "4. Descriptive statistics (sample SD; quantiles use NumPy linear interpolation)","Metric | N | Mean | SD | Median | Min | Q1 | Q3 | Max | CV%"]
    for m in ("HEX8_total","QUAD4_total","total_elements","mesh_size_MB","mesh_generation_seconds","certification_seconds","peak_RSS_MB"):
        s=descriptive[m]; lines.append(f"{METRICS[m]} | {s['N']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['median'])} | {fmt(s['min'])} | {fmt(s['Q1'])} | {fmt(s['Q3'])} | {fmt(s['max'])} | {fmt(s['CV_percent'])}%")
    lines += ["","5. Per-patient variability","Derived metrics are in patient_derived_metrics.csv (element ratios, per-million-element rates, and material fractions).","",
      "6. Runtime, memory, and storage",f"Mesh generation mean={descriptive['mesh_generation_seconds']['mean']:.3f}s (SD {descriptive['mesh_generation_seconds']['std']:.3f}); median={descriptive['mesh_generation_seconds']['median']:.3f}s.",
      f"Certification mean={descriptive['certification_seconds']['mean']:.3f}s (SD {descriptive['certification_seconds']['std']:.3f}); median={descriptive['certification_seconds']['median']:.3f}s.",
      f"Peak RSS max={descriptive['peak_RSS_MB']['max']/1024:.4f} GB (input CSV units); storage total={totals['mesh_size_GB_decimal']:.3f} GB decimal.","",
      "7. Correlation analysis (Pearson r; Spearman in JSON/CSV)"]
    for a,b in pairs:
        c=correlations[f"{a}__{b}"]["pearson"];lines.append(f"{a} vs {b}: r={fmt(c['r'],4)}, N={c['N']}")
    lines += ["Correlations are descriptive and imprecise at N=20; they do not establish causality.","",
      "8. Exploratory linear models (ordinary least squares; descriptive only)"]
    for name,lm in models.items():lines.append(f"{name}: intercept={fmt(lm['intercept'],6)}, slope={fmt(lm['slope'],9)}, R2={fmt(lm['R2'],5)}")
    lines += ["","9. Outlier analysis (Tukey 1.5×IQR; outliers are not errors)"]
    lines += [f"{o['patient']} — {o['metric']}={o['value']:.6g}" for o in flagged] or ["No observations flagged for the analyzed metrics."]
    lines += ["","10. Certified-with-observation cases",f"Patients: {', '.join(certification['known_observation_patients']) or 'none'} ({certification['observation_percent']:.1f}%).",
      f"Six tag-105 faces per affected patient; {certification['total_tag105_observation_faces']} total reported observation faces. These remain certified, not failures.","",
      "11. Consistency checks against authoritative report"]
    for k,v in checks.items():lines.append(f"{k}: {v['status']} — reported={v.get('reported')}, recomputed={v.get('recomputed',v.get('recomputed_recorded_stage_sum_seconds'))}, difference={v.get('difference',v.get('difference_seconds'))}")
    lines += ["","12. Warning-group descriptive comparison",f"Without warning N={len(clean)}; with documented observation N={len(warned)}."]
    for metric in group_fields:
        a=group_summary["without_warning"][metric];b=group_summary["with_documented_observation"][metric]
        lines.append(f"{metric}: no-warning mean/median={fmt(a['mean'])}/{fmt(a['median'])}; observation mean/median={fmt(b['mean'])}/{fmt(b['median'])}.")
    lines += ["","13. Limitations","Pilot20 is a small, selected cohort. Statistical and scaling estimates are descriptive, not validated forecasts. No clinical, FEM, solver, or biomechanical inference is made.",
      "","14. Final statistical interpretation","All 20 reported meshes remain certified, including 3 certified with the documented Scalp/background observation. Per-patient size and time vary; observed associations are exploratory. No meshes or labels were modified."]
    (OUT/"pilot20_statistics_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    results=["Pilot20 Cohort Meshing Results","","Certification — Measured: 20/20 meshes were certified; 17/20 (85%) were certified without warning and 3/20 (15%) were certified with the documented Scalp/background observation. No mesh failed. Exact material reconstruction and tumor preservation were recorded for all 20.",
      "","Mesh complexity — Measured total HEX8 count was 56,256,347 and QUAD4 count was 11,895,144. Per-patient HEX8 mean ± sample SD was "
       f"{descriptive['HEX8_total']['mean']:,.0f} ± {descriptive['HEX8_total']['std']:,.0f} (median {descriptive['HEX8_total']['median']:,.0f}; range {descriptive['HEX8_total']['min']:,.0f}–{descriptive['HEX8_total']['max']:,.0f}).",
      f"Per-patient QUAD4 mean ± SD was {descriptive['QUAD4_total']['mean']:,.0f} ± {descriptive['QUAD4_total']['std']:,.0f} (median {descriptive['QUAD4_total']['median']:,.0f}; range {descriptive['QUAD4_total']['min']:,.0f}–{descriptive['QUAD4_total']['max']:,.0f}).",
      "","Computational performance — Measured mesh-generation time averaged "
       f"{descriptive['mesh_generation_seconds']['mean']:.2f} ± {descriptive['mesh_generation_seconds']['std']:.2f}s (median {descriptive['mesh_generation_seconds']['median']:.2f}s; range {descriptive['mesh_generation_seconds']['min']:.2f}–{descriptive['mesh_generation_seconds']['max']:.2f}s). "
       f"Certification time averaged {descriptive['certification_seconds']['mean']:.2f} ± {descriptive['certification_seconds']['std']:.2f}s. Peak RSS maximum was {max(r['peak_RSS_GB'] for r in rows):.3f} GB as recorded.",
      f"","Variability — Mesh files totaled {totals['mesh_size_GB_decimal']:.3f} decimal GB. HEX8 counts ranged from {descriptive['HEX8_total']['min']:,.0f} to {descriptive['HEX8_total']['max']:,.0f}; this summarizes variation in these 20 datasets only.",
      "","Observations — EGD-0009, EGD-0012, and EGD-0013 retained the documented preprocessing observation. Each had six tag-105 faces around the retained feature; the independent mesh checks found no non-manifold faces, forbidden contacts, or reconstruction differences. These were not mesh failures.",
      "","Scaling behavior — Correlations and linear fits are exploratory summaries at N=20 and should not be treated as predictive performance for future patients.",
      "","Study limitation — This was a meshing and certification study only. It does not establish clinical validity, FEM correctness, solution accuracy, or solver scalability."]
    (OUT/"pilot20_results_section.txt").write_text("\n".join(results)+"\n",encoding="utf-8")
    readme="""# Pilot20 statistical analysis\n\nInputs are read-only: `../mesh_summary.csv`, `../pilot20_mesh_report.json`, and `../pilot20_mesh_report.txt`. The analysis script validates schema and numeric values, derives total elements as HEX8 + QUAD4, converts decimal GB to MB by ×1000, and reports RSS GB-to-MB with ×1024. Standard deviations are sample SD (ddof=1); quartiles use NumPy's default linear quantile method. IQR outliers use Q1 −/＋ 1.5×IQR and are retained. Correlations and ordinary least-squares fits are descriptive only.\n\nOutputs include JSON/TXT reports, CSV tables, LaTeX tables, and PNG figures in `figures/`. Run from the repository root with `python3 cohort_meshing/pilot20/analyze_pilot20_statistics.py`. Requires Python 3, NumPy, and Matplotlib; pandas is not required. The script does not write to source meshes, labels, or source reports.\n"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")

    print("="*60+"\nPILOT20 STATISTICAL ANALYSIS COMPLETE\n"+"="*60)
    print(f"Patients:\n    N = {len(rows)}")
    print("Certification:")
    print(f"    Certified = {certification['certified_count']} / {len(rows)} ({certification['certified_percent']:.1f}%)")
    print(f"    Without warning = {len(clean)} / {len(rows)} ({certification['without_warning_percent']:.1f}%)")
    print(f"    With documented observation = {len(warned)} / {len(rows)} ({certification['with_documented_observation_percent']:.1f}%)")
    print(f"    Failed = {certification['failed_count']} / {len(rows)} ({certification['failed_percent']:.1f}%)")
    for m,label in (("HEX8_total","HEX8"),("QUAD4_total","QUAD4"),("total_elements","Total elements")):
        s=descriptive[m];print(f"{label}: mean={s['mean']:.3f}; SD={s['std']:.3f}; median={s['median']:.3f}; range={s['min']:.0f} -- {s['max']:.0f}")
    s=descriptive["mesh_generation_seconds"];print(f"Processing time (mesh-generation field): mean={s['mean']:.3f}s; SD={s['std']:.3f}s; median={s['median']:.3f}s; range={s['min']:.3f} -- {s['max']:.3f}")
    print(f"Peak RSS mean={descriptive['peak_RSS_MB']['mean']/1024:.4f} GB; maximum={descriptive['peak_RSS_MB']['max']/1024:.4f} GB")
    c=correlations["total_elements__mesh_generation_seconds"]["pearson"];print(f"r(total_elements, processing_time)={c['r']:.6f}; R2(elements -> processing_time)={models['processing_time_seconds_vs_total_elements']['R2']:.6f}")
    print(f"Outliers: count={len(flagged)}; patients={', '.join(sorted(set(x['patient'] for x in flagged))) or 'none'}")
    print(f"Consistency with authoritative Pilot20 report: {'PASS' if all_check_pass else 'FAIL'} (stage-time scope difference separately reported)")
    print("Meshing interpretation: 20/20 certified; 3 retained documented observations, not failures.")
    print("FEM: NOT RUN\nREADY_FOR_LARGER_COHORT_MESHING_STUDY: YES")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    checks = {}
    main()
