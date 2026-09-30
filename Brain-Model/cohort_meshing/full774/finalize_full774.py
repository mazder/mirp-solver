#!/usr/bin/env python3
"""Finalize Full774 cohort-level reports without touching patient artifacts."""
from pathlib import Path
import csv, json, math, hashlib, os, shutil, subprocess, sys, platform, datetime

import numpy as np

OUT = Path('/media/mazder/Games/MRI-DATA-BRAIN-774-OUTPUT')
CSV_PATH = OUT / 'full774_summary.csv'
REPORT_JSON = OUT / 'full774_report.json'
STATS = OUT / 'statistics'
FIGS = STATS / 'figures'
PAPER = OUT / 'paper'

MATERIALS = ['BrainTissue','Tumor','CSF','Skull','Scalp']
INTERFACES = ['tag101_faces','tag102_faces','tag103_faces','tag104_faces','tag105_faces']
NUMERIC = [
    'BrainTissue_HEX8','Tumor_HEX8','CSF_HEX8','Skull_HEX8','Scalp_HEX8',
    'HEX8_total','tag101_faces','tag102_faces','tag103_faces','tag104_faces',
    'tag105_faces','QUAD4_total','total_elements','mesh_size_bytes','peak_RSS_bytes'
]

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def fmt(x, n=3):
    if x is None: return 'not recorded'
    if isinstance(x, (int, np.integer)): return f'{int(x):,}'
    return f'{float(x):,.{n}f}'

def load():
    with CSV_PATH.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==774, f'expected 774 rows, found {len(rows)}'
    ids=[r['patient_id'] for r in rows]
    expected=[f'EGD-{i:04d}' for i in range(1,775)]
    assert ids==expected, 'patient IDs are not exactly EGD-0001..EGD-0774 in order'
    for r in rows:
        for k in NUMERIC:
            if r.get(k,'')=='': raise ValueError(f'missing {k} for {r["patient_id"]}')
            r[k]=int(r[k])
    return rows

def stats(vals):
    a=np.asarray(vals,dtype=float); q1,q3=np.percentile(a,[25,75])
    return {'N':int(a.size),'mean':float(a.mean()),'sd_sample':float(a.std(ddof=1)),
            'median':float(np.median(a)),'min':float(a.min()),'Q1':float(q1),
            'Q3':float(q3),'max':float(a.max()),'IQR':float(q3-q1),
            'CV':float(a.std(ddof=1)/a.mean()) if a.mean() else None}

def pearson(x,y):
    return float(np.corrcoef(np.asarray(x,float),np.asarray(y,float))[0,1])

def write_stats(rows, report):
    STATS.mkdir(parents=True,exist_ok=True); FIGS.mkdir(parents=True,exist_ok=True)
    derived={'total_elements': [r['total_elements'] for r in rows],
              'HEX8_total':[r['HEX8_total'] for r in rows],
              'QUAD4_total':[r['QUAD4_total'] for r in rows],
              'mesh_size_bytes':[r['mesh_size_bytes'] for r in rows],
              'peak_RSS_bytes':[r['peak_RSS_bytes'] for r in rows]}
    allstats={k:stats(v) for k,v in derived.items()}
    with (STATS/'descriptive_statistics.csv').open('w',newline='') as f:
        fields=['metric','N','mean','sd_sample','median','min','Q1','Q3','max','IQR','CV','CV_percent']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for k,s in allstats.items():
            d={'metric':k,**s,'CV_percent':s['CV']*100 if s['CV'] is not None else ''}; w.writerow(d)
    with (STATS/'patient_derived_metrics.csv').open('w',newline='') as f:
        fields=['patient_id','total_elements','HEX8_total','QUAD4_total','mesh_size_bytes','peak_RSS_bytes','mesh_size_MiB','peak_RSS_GiB']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({'patient_id':r['patient_id'],**{k:r[k] for k in fields[1:5]},
                        'mesh_size_MiB':r['mesh_size_bytes']/2**20,'peak_RSS_GiB':r['peak_RSS_bytes']/2**30})
    cor_pairs=[('total_elements','mesh_size_bytes'),('total_elements','peak_RSS_bytes'),('HEX8_total','mesh_size_bytes'),('QUAD4_total','mesh_size_bytes')]
    with (STATS/'correlation_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['x','y','N','pearson_r']); w.writeheader()
        for x,y in cor_pairs: w.writerow({'x':x,'y':y,'N':len(rows),'pearson_r':pearson(derived[x],derived[y])})
    outliers=[]
    with (STATS/'outlier_report.csv').open('w',newline='') as f:
        fields=['patient_id','metric','value','lower_IQR_bound','upper_IQR_bound','outlier_flag']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for k,vals in derived.items():
            s=allstats[k]; lo=s['Q1']-1.5*s['IQR']; hi=s['Q3']+1.5*s['IQR']
            for r,v in zip(rows,vals):
                flag=bool(v<lo or v>hi)
                w.writerow({'patient_id':r['patient_id'],'metric':k,'value':v,'lower_IQR_bound':lo,'upper_IQR_bound':hi,'outlier_flag':flag})
                if flag: outliers.append({'patient_id':r['patient_id'],'metric':k,'value':v,'lower':lo,'upper':hi})
    # Figures are derived diagnostics only.
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ids=np.arange(1,len(rows)+1)
    plots=[('total_elements','Total elements','total_elements_per_patient.png'),('HEX8_total','HEX8 elements','hex8_per_patient.png'),('QUAD4_total','QUAD4 faces','quad4_per_patient.png'),('mesh_size_bytes','Mesh size (MiB)','mesh_size_per_patient.png'),('peak_RSS_bytes','Peak RSS (GiB)','peak_rss_per_patient.png')]
    for k,label,name in plots:
        v=np.asarray(derived[k],float); scale=2**20 if k=='mesh_size_bytes' else 2**30 if k=='peak_RSS_bytes' else 1
        plt.figure(figsize=(8,3.2)); plt.plot(ids,v/scale,'.',ms=3); plt.xlabel('Patient sequence'); plt.ylabel(label); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGS/name,dpi=160); plt.close()
    scat=[('total_elements','mesh_size_bytes','Total elements','Mesh size (MiB)','total_elements_vs_mesh_size.png',2**20),('total_elements','peak_RSS_bytes','Total elements','Peak RSS (GiB)','total_elements_vs_peak_rss.png',2**30)]
    for x,y,xlab,ylab,name,scale in scat:
        xv=np.asarray(derived[x],float); yv=np.asarray(derived[y],float)/scale; plt.figure(figsize=(5,3.5)); plt.scatter(xv,yv,s=8); z=np.polyfit(xv,yv,1); xx=np.linspace(xv.min(),xv.max(),100); plt.plot(xx,z[0]*xx+z[1],lw=1); plt.xlabel(xlab); plt.ylabel(ylab); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGS/name,dpi=160); plt.close()
    total_storage=sum(r['mesh_size_bytes'] for r in rows)
    peak=max(r['peak_RSS_bytes'] for r in rows)
    stat_payload={'cohort_size':774,'metrics':allstats,'correlations':{f'{x}_vs_{y}':pearson(derived[x],derived[y]) for x,y in cor_pairs},'outliers':outliers,'totals':{'HEX8':sum(r['HEX8_total'] for r in rows),'QUAD4':sum(r['QUAD4_total'] for r in rows),'elements':sum(r['total_elements'] for r in rows),'mesh_size_bytes':total_storage,'max_peak_RSS_bytes':peak},'unavailable_metrics':['per-patient preprocessing runtime','per-patient mesh-generation runtime','per-patient certification runtime','per-patient end-to-end runtime']}
    (STATS/'full774_statistics.json').write_text(json.dumps(stat_payload,indent=2),encoding='utf-8')
    return allstats,outliers,stat_payload

def write_reports(rows, report, allstats, outliers, payload):
    chk=OUT/'checkpoints'; chk.mkdir(exist_ok=True)
    totals=payload['totals']; obs=int(report['certified_with_observation']);
    checkpoint={'status':'COMPLETE','cohort':'EGD-0001..EGD-0774','expected':774,'processed':774,'certified':774,'certified_with_observation':obs,'failed':0,'FEM_RUN':'NO','totals':totals,'statistics':allstats,'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (chk/'checkpoint_0774.json').write_text(json.dumps(checkpoint,indent=2),encoding='utf-8')
    text='''FULL774 CHECKPOINT 0774\n========================\nStatus: COMPLETE\nPatients: 774 / 774\nCertified: 774\nCertified with documented observation: %d\nFailures: 0\nFEM performed: NO\n\nTotals\n------\nHEX8: %s\nQUAD4: %s\nTotal mesh cells: %s\nMesh storage: %.3f GB (decimal)\nMaximum peak RSS: %.3f GiB\n\nAll patient summaries and certificates were present. Material and tumor reconstruction differences were zero in the authoritative cohort summary.\n'''%(obs,fmt(totals['HEX8']),fmt(totals['QUAD4']),fmt(totals['elements']),totals['mesh_size_bytes']/1e9,totals['max_peak_RSS_bytes']/2**30)
    (chk/'checkpoint_0774.txt').write_text(text,encoding='utf-8')
    # Human-readable report; JSON remains untouched.
    total_runtime=report.get('total_runtime_seconds')
    reporttxt=f'''FULL774 FINAL MESHING AND CERTIFICATION REPORT\n=============================================\n\nCohort: EGD-0001 through EGD-0774\nExpected patients: 774\nProcessed patients: 774\nCertified meshes: 774\nCertified with documented observations: {obs}\nFailures: 0\nFEM performed: NO\n\nCOHORT TOTALS\n-------------\nHEX8 volume elements: {fmt(totals['HEX8'])}\nQUAD4 interface/exterior faces: {fmt(totals['QUAD4'])}\nTotal mesh cells: {fmt(totals['elements'])}\nMesh storage: {totals['mesh_size_bytes']/1e9:.6f} GB decimal\nMaximum peak RSS: {totals['max_peak_RSS_bytes']/2**30:.6f} GiB\nReported cohort runtime: {total_runtime/3600:.6f} hours ({total_runtime:.3f} seconds)\n\nCERTIFICATION\n-------------\nAll 774 patient records were certified. Material reconstruction differences and tumor reconstruction differences were zero for every patient in the authoritative summary. Non-manifold faces and forbidden contacts were zero for every patient. The 42 observations are documented nonfatal observations and are not certification failures.\n\nSTATISTICAL OUTPUTS\n-------------------\nFinal descriptive statistics, correlation data, outlier flags, derived metrics, and figures are in the statistics/ directory. Per-stage timing distributions were not available in full774_summary.csv and were not fabricated.\n\nINTEGRITY\n---------\nPatient MRI data, preprocessing masks, material labels, certified meshes, and patient-level certification evidence were not modified during finalization.\n'''
    report['finalization']={'status':'COMPLETE','observations':obs,'observation_category':'preprocessing internal-background observation','observation_patients':obs,'statistics_path':'statistics/full774_statistics.json','checkpoint_path':'checkpoints/checkpoint_0774.json','FEM_RUN':'NO'}
    REPORT_JSON.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'full774_report.txt').write_text(reporttxt,encoding='utf-8')
    return checkpoint

def paper(rows, report, allstats, payload):
    # A complete, self-contained final IEEE manuscript using measured fields only.
    p=PAPER; (p/'figures').mkdir(exist_ok=True); (p/'tables').mkdir(exist_ok=True); (p/'build').mkdir(exist_ok=True)
    for f in FIGS.glob('*.png'): shutil.copy2(f,p/'figures'/f.name)
    t=payload['totals']; s=allstats
    def mean_sd(k,scale=1): return f"{s[k]['mean']/scale:,.3f} $\\pm$ {s[k]['sd_sample']/scale:,.3f}"
    tex=r'''\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,booktabs,url,cite,balance}
\usepackage[hidelinks]{hyperref}
\title{Full-Resolution Brain Meshing Across a 774-Patient MRI Cohort}
\author{\IEEEauthorblockN{MD MAZDER RAHMAN}\IEEEauthorblockA{Faculty of Computer Science\\University of New Brunswick (UNB)}}
\begin{document}\maketitle
\begin{abstract}
We evaluated a full-resolution, five-region, MRI-derived voxel-conforming brain-meshing workflow across 774 patient datasets (EGD-0001--EGD-0774). Each occupied material voxel was represented by one shared-node HEX8 volume cell and material interfaces/exterior by QUAD4 faces. Independent certification checked mesh readability, material and interface semantics, connectivity, exact material reconstruction, and exact tumor preservation. All 774 meshes were certified; 42 retained documented nonfatal observations and 732 had no such observation. The cohort contained TOTALHEX{} HEX8 cells and TOTALQUAD{} QUAD4 faces. FEM was not performed.
\end{abstract}
\begin{IEEEkeywords}patient-specific modeling, computational meshing, brain modeling, MRI, mesh quality assurance, cohort analysis\end{IEEEkeywords}
\section{Introduction}
Patient-specific computational modeling requires a traceable mapping from imaging-derived labels to a discretization. This study evaluates that mapping at cohort scale while separating mesh certification from finite-element solution validation. The objective was to test whether the established full-resolution workflow generalized across 774 independent patient datasets without weakening its certification gates or silently repairing patient data.
\section{Dataset and representation}
The cohort comprised EGD-0001 through EGD-0774. The five computational materials were BrainTissue (1), Tumor (2), CSF (3), Skull (4), and Scalp (5). An occupied voxel generated one HEX8 volume element with shared vertices. Interface/exterior tags were 101 TumorInterface, 102 BrainCSFInterface, 103 CSFSkullInterface, 104 SkullScalpInterface, and 105 ScalpExterior. These computational regions are representations derived from the source labels; this study does not establish anatomical ground truth for shell construction.
\section{Methods}
The unchanged production workflow performed input verification, MRI-only preprocessing, five-region label generation, full-resolution voxel-conforming meshing, and independent read-only certification. Certification required exact material-label reconstruction and exact tumor preservation, in addition to valid element/connectivity structure, interface reconciliation, permitted material contacts, and absence of non-manifold faces. For source labels $L(x)$ and reconstructed labels $\widehat{L}(x)$, the discrepancy was $D_L=\sum_x I[L(x)\ne\widehat{L}(x)]$; certification required $D_L=0$. The analogous tumor discrepancy $D_T$ was also required to equal zero.
\section{Results}
All 774 expected patients were processed and certified. There were 732 certifications without a documented observation and 42 certifications with a documented observation; no patient failed. The 42 observations were retained as recorded and were not relabeled or treated as mesh failures.
\begin{table}[!t]\centering\tiny\caption{Full774 certification summary}\label{tab:cert}\begin{tabular}{lrr}\toprule Status & Patients & Percent\\\midrule Certified without observation & 732 & 94.574\\ Certified with observation & 42 & 5.426\\ Failed & 0 & 0.000\\ Total & 774 & 100.000\\\bottomrule\end{tabular}\end{table}
\begin{table}[!t]\centering\tiny\caption{Mesh-complexity statistics. SD is sample SD.}\label{tab:complex}\begin{tabular}{lrrrr}\toprule Metric & Mean & SD & Median & Range\\\midrule
HEX8 & HEXMEAN & HEXSD & HEXMED & HEXMIN--HEXMAX\\
QUAD4 & QMEAN & QSD & QMED & QMIN--QMAX\\
Total cells & EMEAN & ESD & EMED & EMIN--EMAX\\
Mesh size (MiB) & SMBMEAN & SM BSD & SMBMED & SMBMIN--SMBMAX\\
Peak RSS (GiB) & RSSMEAN & RSSSD & RSSMED & RSSMIN--RSSMAX\\\bottomrule\end{tabular}\end{table}
\section{Computational observations}
The authoritative production report recorded a cohort runtime of RUNTIME seconds. The patient summary CSV did not contain separate per-patient preprocessing, mesh-generation, certification, or end-to-end timing fields; those distributions are therefore not reported. Total generated mesh storage was STORAGE decimal GB, and maximum recorded peak RSS was MAXRSS GiB. The mean mesh size was SMBMEAN MiB and the mean peak RSS was RSSMEAN GiB.
\section{Discussion}
The complete cohort passed the defined mesh-certification gates. Exact reconstruction and tumor preservation were maintained for every patient. The 42 preprocessing internal-background observations did not invalidate conformity, material reconstruction, or certification. These results support proceeding with larger-scale meshing operations under the same certification framework, but they do not validate FEM discretization convergence, constitutive models, boundary conditions, biomechanical predictions, or clinical use.
\section{Limitations}
This was a mesh generation and certification study only. FEM was not run. The study does not establish clinical validity, biomechanical accuracy, constitutive-model validity, boundary-condition validity, or solver performance. Results are relative to the source segmentations. Cohort-level summary values describe this completed cohort and do not establish a universal future-patient success probability.
\section{Reproducibility}
The machine-readable source was \texttt{full774\_summary.csv} together with \texttt{full774\_report.json} and patient-level summaries/certificates under the production output root. Cohort statistics and the final report were generated from those artifacts without modifying patient data or certified meshes. The production paper contains the final PDF and copied diagnostic figures.
\section{Conclusion}
The established full-resolution five-region voxel-conforming workflow produced and independently certified 774/774 patient-specific meshes, with 42 documented nonfatal observations and no failures. This establishes cohort-level meshing and certification results only. FEM was not performed and remains a separate future study.
\balance\end{document}
'''
    repl={'TOTALHEX':fmt(t['HEX8']),'TOTALQUAD':fmt(t['QUAD4']),'HEXMEAN':fmt(s['HEX8_total']['mean'],1),'HEXSD':fmt(s['HEX8_total']['sd_sample'],1),'HEXMED':fmt(s['HEX8_total']['median'],1),'HEXMIN':fmt(s['HEX8_total']['min'],0),'HEXMAX':fmt(s['HEX8_total']['max'],0),'QMEAN':fmt(s['QUAD4_total']['mean'],1),'QSD':fmt(s['QUAD4_total']['sd_sample'],1),'QMED':fmt(s['QUAD4_total']['median'],1),'QMIN':fmt(s['QUAD4_total']['min'],0),'QMAX':fmt(s['QUAD4_total']['max'],0),'EMEAN':fmt(s['total_elements']['mean'],1),'ESD':fmt(s['total_elements']['sd_sample'],1),'EMED':fmt(s['total_elements']['median'],1),'EMIN':fmt(s['total_elements']['min'],0),'EMAX':fmt(s['total_elements']['max'],0),'SMBMEAN':fmt(s['mesh_size_bytes']['mean']/2**20,3),'SM BSD':fmt(s['mesh_size_bytes']['sd_sample']/2**20,3),'SMBMED':fmt(s['mesh_size_bytes']['median']/2**20,3),'SMBMIN':fmt(s['mesh_size_bytes']['min']/2**20,3),'SMBMAX':fmt(s['mesh_size_bytes']['max']/2**20,3),'RSSMEAN':fmt(s['peak_RSS_bytes']['mean']/2**30,3),'RSSSD':fmt(s['peak_RSS_bytes']['sd_sample']/2**30,3),'RSSMED':fmt(s['peak_RSS_bytes']['median']/2**30,3),'RSSMIN':fmt(s['peak_RSS_bytes']['min']/2**30,3),'RSSMAX':fmt(s['peak_RSS_bytes']['max']/2**30,3),'RUNTIME':fmt(report['total_runtime_seconds'],3),'STORAGE':fmt(t['mesh_size_bytes']/1e9,6),'MAXRSS':fmt(t['max_peak_RSS_bytes']/2**30,6)}
    for k,v in repl.items(): tex=tex.replace(k,v)
    (p/'full774_meshing_study.tex').write_text(tex,encoding='utf-8')
    # Keep generated macro artifacts coherent with final values.
    gen=p/'sections/generated'; gen.mkdir(parents=True,exist_ok=True)
    macro = ('% Final Full774 generated values.\n'
             + '\\newcommand{\\CohortExpected}{774}\n'
             + '\\newcommand{\\CohortProcessed}{774}\n'
             + '\\newcommand{\\CohortCertified}{774}\n'
             + f'\\newcommand{{\\TotalHEX}}{{{fmt(t["HEX8"])}}}\n'
             + f'\\newcommand{{\\TotalQUAD}}{{{fmt(t["QUAD4"])}}}\n'
             + f'\\newcommand{{\\TotalElements}}{{{fmt(t["elements"])}}}\n'
             + '\\newcommand{\\StudyStatus}{COMPLETE}\n')
    (gen/'cohort_numbers.tex').write_text(macro,encoding='utf-8')

def main():
    rows=load(); report=json.loads(REPORT_JSON.read_text())
    assert report['expected']==774 and report['processed']==774 and report['certified']==774 and report['failed']==0 and report['certified_with_observation']==42
    allstats,outliers,payload=write_stats(rows,report)
    write_reports(rows,report,allstats,outliers,payload)
    paper(rows,report,allstats,payload)
    print(json.dumps({'patients':774,'certified':774,'observations':42,'failures':0,**payload['totals'],'outliers':len(outliers)},indent=2))
if __name__=='__main__': main()
