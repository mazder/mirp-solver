#!/usr/bin/env python3
"""Sequential, resumable full-resolution Pilot20 voxel-HEX8 generation/certification.

Preprocessing labels are strictly read-only. Mesh construction reuses the Phase-4B
voxel conventions; certification parses the written Gmsh file independently.
No FEM operations are performed.
"""
from __future__ import annotations
import csv, gc, hashlib, json, math, os, resource, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PILOT = ROOT / "cohort_meshing/pilot20"
PRE = PILOT / "preprocessing"
OUT = PILOT / "meshes"
SUMMARY = PILOT / "mesh_summary.csv"
MATERIALS = {1:"BrainTissue", 2:"Tumor", 3:"CSF", 4:"Skull", 5:"Scalp"}
PAIR_TAG = {(1,2):101, (1,3):102, (3,4):103, (4,5):104}
TAG_NAMES = {101:"TumorInterface",102:"BrainCSFInterface",103:"CSFSkullInterface",104:"SkullScalpInterface",105:"ScalpExterior"}
WARNING_CASES = {"EGD-0009", "EGD-0012", "EGD-0013"}
SIX = ndimage.generate_binary_structure(3,1)
HEX_OFFSETS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.int64)
FACE_OFFSETS = {
  0: np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]]),
  1: np.array([[0,0,0],[0,0,1],[1,0,1],[1,0,0]]),
  2: np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def rss_gb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/(1024*1024)

def mem_current_gb() -> float:
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return float(line.split()[1])/1024**2
    except Exception: pass
    return rss_gb()

def expected_interfaces(labels: np.ndarray) -> dict[str,int]:
    out={str(i):0 for i in range(101,106)}
    for ax in range(3):
        lo=[slice(None)]*3; hi=[slice(None)]*3; lo[ax]=slice(None,-1); hi[ax]=slice(1,None)
        a,b=labels[tuple(lo)],labels[tuple(hi)]
        for pair,tag in PAIR_TAG.items(): out[str(tag)]+=int(np.count_nonzero(((a==pair[0])&(b==pair[1]))|((a==pair[1])&(b==pair[0]))))
        out['105']+=int(np.count_nonzero(((a==5)&(b==0))|((a==0)&(b==5))))
    # Non-background labels must be away from array edges due to preprocessing padding.
    for ax in range(3):
        for side in (0,-1):
            sl=[slice(None)]*3; sl[ax]=side
            out['105']+=int(np.count_nonzero(labels[tuple(sl)]==5))
    return out

def label_topology(labels: np.ndarray) -> dict:
    bg=labels==0; bl,n=ndimage.label(bg,structure=SIX)
    boundary=np.zeros(bg.shape,bool); boundary[[0,-1],:,:]=1; boundary[:,[0,-1],:]=1; boundary[:,:,[0,-1]]=1
    ext=set(np.unique(bl[boundary]).tolist())-{0}; internal=bg & ~np.isin(bl,list(ext))
    il,inc=ndimage.label(internal,structure=SIX)
    vl,vn=ndimage.label(labels>0,structure=SIX)
    return {"internal_background_components":int(inc),"internal_background_voxels":int(internal.sum()),
            "volume_components_6":int(vn),"background_labels":bl,"background_external_ids":ext,
            "internal_background_labels":il}

def read_msh(path: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,dict]:
    """Independent minimal Gmsh 2.2 ASCII reader; returns points, hexes/tags, quads/tags, names."""
    with path.open('r',encoding='ascii') as f:
        lines=iter(f); fmt=None; names={}; points=None; hexrows=None; hextags=None; quadrows=None; quadtags=None; nh=nq=0
        for line in lines:
            s=line.strip()
            if s=='$MeshFormat': fmt=next(lines).strip(); next(lines)
            elif s=='$PhysicalNames':
                n=int(next(lines))
                for _ in range(n):
                    z=next(lines).strip().split(maxsplit=2); names[(int(z[0]),int(z[1]))]=z[2].strip('"')
                next(lines)
            elif s=='$Nodes':
                n=int(next(lines)); points=np.empty((n,3),dtype=np.float64)
                for idx in range(n):
                    z=next(lines).split()
                    if int(z[0]) != idx+1: raise ValueError('node IDs not contiguous/ordered')
                    points[idx]=[float(z[1]),float(z[2]),float(z[3])]
                next(lines)
            elif s=='$Elements':
                n=int(next(lines))
                hexrows=np.empty((n,8),dtype=np.int64);hextags=np.empty(n,dtype=np.int32)
                quadrows=np.empty((n,4),dtype=np.int64);quadtags=np.empty(n,dtype=np.int32)
                for _ in range(n):
                    z=next(lines).split(); typ=int(z[1]); nt=int(z[2]); tags=list(map(int,z[3:3+nt])); conn=np.asarray(z[3+nt:],dtype=np.int64)-1
                    if typ==5:
                        if conn.size!=8: raise ValueError('invalid HEX8 connectivity width')
                        hexrows[nh]=conn;hextags[nh]=tags[0] if tags else 0;nh+=1
                    elif typ==3:
                        if conn.size!=4: raise ValueError('invalid QUAD4 connectivity width')
                        quadrows[nq]=conn;quadtags[nq]=tags[0] if tags else 0;nq+=1
                    else: raise ValueError(f'unexpected mesh element type {typ}')
                next(lines)
        if fmt is None or points is None: raise ValueError('missing Gmsh sections')
    return points,hexrows[:nh],hextags[:nh],quadrows[:nq],quadtags[:nq],{"format":fmt,"physical_names":names}

def cert_mesh(path: Path, labels: np.ndarray, affine: np.ndarray, expected_hash: str) -> dict:
    started=time.monotonic();print(f"  CERT: parse MSH {path.stat().st_size/1e6:.1f} MB",flush=True);points,hexes,htags,quads,qtags,meta=read_msh(path)
    print(f"  CERT: parsed nodes={len(points):,} HEX8={len(hexes):,} QUAD4={len(quads):,}",flush=True)
    shape=labels.shape; inv=np.linalg.inv(affine); det=float(np.linalg.det(affine[:3,:3])); spacing=np.linalg.norm(affine[:3,:3],axis=0)
    result={"mesh_sha256":sha(path),"mesh_sha256_match_written":sha(path)==expected_hash,
            "mesh_format":meta['format'],"vertices":int(len(points)),"hex8":int(len(hexes)),"quad4":int(len(quads)),
            "hex8_by_material":{str(m):int(np.count_nonzero(htags==m)) for m in range(1,6)},
            "quad4_by_tag":{str(t):int(np.count_nonzero(qtags==t)) for t in range(101,106)},
            "physical_names":{f"{d}:{t}":n for (d,t),n in meta['physical_names'].items()}}
    valid_ids=bool(hexes.size and quads.size and hexes.min()>=0 and quads.min()>=0 and max(int(hexes.max()),int(quads.max()))<len(points))
    result['connectivity_indices_valid']=valid_ids
    if not valid_ids: raise ValueError('out-of-range or empty mesh connectivity')
    # Independently map every disk node to the affine voxel-corner lattice.
    node_lattice=np.empty((len(points),3),dtype=np.int32)
    max_coord_err=0.0
    for s in range(0,len(points),250000):
        e=min(s+250000,len(points)); ijk=nib.affines.apply_affine(inv,points[s:e])+0.5
        near=np.rint(ijk); max_coord_err=max(max_coord_err,float(np.max(np.abs(ijk-near))))
        node_lattice[s:e]=near.astype(np.int32)
    result['max_node_coordinate_lattice_error']=max_coord_err
    node_shape=np.asarray(shape,dtype=np.int64)+1
    in_grid=bool(np.all(node_lattice>=0) and np.all(node_lattice<node_shape))
    node_key=np.ravel_multi_index(node_lattice.T,tuple(node_shape)) if in_grid else np.empty(0,dtype=np.int64)
    result['all_nodes_on_expected_affine_lattice']=in_grid and max_coord_err<2e-7
    result['duplicate_mesh_nodes']=int(len(node_key)-len(np.unique(node_key))) if in_grid else -1
    print("  CERT: validating/reconstructing HEX8",flush=True)
    # Reconstruct every cell's source voxel from the physical element centroid.
    reconstructed=np.zeros(shape,dtype=np.uint8); seen=np.zeros(shape,dtype=bool)
    cell_indices=np.full((len(hexes),3),-999,dtype=np.int32); max_center_err=0.0; min_jac=math.inf; max_jac=-math.inf; min_scaled=math.inf; max_scaled=-math.inf
    inverted=zero_jac=0; min_edge=math.inf; max_edge=0.0
    edge_pairs=np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
    expected_hex_counts={str(m):int(np.count_nonzero(labels==m)) for m in range(1,6)}
    for s in range(0,len(hexes),100000):
        e=min(s+100000,len(hexes)); hc=hexes[s:e]
        xyz=points[hc]; cent=xyz.mean(axis=1); vox=nib.affines.apply_affine(inv,cent)
        near=np.rint(vox); max_center_err=max(max_center_err,float(np.max(np.abs(vox-near))))
        idx=near.astype(np.int32); cell_indices[s:e]=idx
        ok=np.all((idx>=0)&(idx<np.asarray(shape)),axis=1)
        if not np.all(ok): continue
        conn_lattice=node_lattice[hc].astype(np.int64)
        want=idx[:,None,:]+HEX_OFFSETS[None,:,:]
        key_actual=np.sum(conn_lattice*np.array([node_shape[1]*node_shape[2],node_shape[2],1]),axis=2)
        key_want=np.sum(want*np.array([node_shape[1]*node_shape[2],node_shape[2],1]),axis=2)
        if np.any(np.sort(key_actual,axis=1)!=np.sort(key_want,axis=1)):
            raise ValueError('HEX8 corners do not map to the expected structured voxel cell')
        flat=np.ravel_multi_index(idx.T,shape)
        if np.any(seen.ravel()[flat]): raise ValueError('duplicate HEX8 cell maps to the same image voxel')
        seen.ravel()[flat]=True
        actual=labels[tuple(idx.T)]
        if np.any(actual!=htags[s:e]): raise ValueError('HEX8 material tag disagrees with source label voxel')
        reconstructed[tuple(idx.T)]=htags[s:e]
        v1=xyz[:,1]-xyz[:,0]; v2=xyz[:,3]-xyz[:,0]; v3=xyz[:,4]-xyz[:,0]
        jac=np.einsum('ij,ij->i',np.cross(v1,v2),v3)
        min_jac=min(min_jac,float(jac.min())); max_jac=max(max_jac,float(jac.max()))
        inverted+=int(np.count_nonzero(jac<0));zero_jac+=int(np.count_nonzero(jac==0))
        denom=np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)*np.linalg.norm(v3,axis=1)
        sj=jac/denom
        min_scaled=min(min_scaled,float(sj.min())); max_scaled=max(max_scaled,float(sj.max()))
        el=np.linalg.norm(xyz[:,edge_pairs[:,0]]-xyz[:,edge_pairs[:,1]],axis=2)
        min_edge=min(min_edge,float(el.min()));max_edge=max(max_edge,float(el.max()))
    diff=int(np.count_nonzero(reconstructed!=labels))
    tumor_src=labels==2; tumor_rec=reconstructed==2; tp=int(np.count_nonzero(tumor_src&tumor_rec)); fp=int(np.count_nonzero(tumor_rec&~tumor_src)); fn=int(np.count_nonzero(tumor_src&~tumor_rec))
    result.update({"max_element_center_lattice_error":max_center_err,"image_reconstruction_differing_voxels":diff,
       "occupied_source_voxels":int(np.count_nonzero(labels)),"occupied_reconstructed_voxels":int(np.count_nonzero(reconstructed)),
       "inverted_hex8":inverted,"zero_jacobian_hex8":zero_jac,
       "minimum_edge_length_mm":min_edge,"maximum_edge_length_mm":max_edge,
       "maximum_edge_ratio_aspect_metric":max_edge/min_edge if min_edge>0 else None,
       "minimum_signed_jacobian":min_jac,"maximum_signed_jacobian":max_jac,
       "minimum_hex_volume_mm3":float(abs(det)),"maximum_hex_volume_mm3":float(abs(det)),
       "minimum_scaled_jacobian":min_scaled,"maximum_scaled_jacobian":max_scaled,
       "tumor_false_positive":fp,"tumor_false_negative":fn,"tumor_differing_voxels":fp+fn,
       "tumor_dice":float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 1.0,
       "tumor_jaccard":float(tp/(tp+fp+fn)) if (tp+fp+fn) else 1.0})
    result['invalid_hex8']=int(np.count_nonzero(np.any((cell_indices<0)|(cell_indices>=np.asarray(shape)),axis=1)))
    result['duplicate_hex8']=int(len(hexes)-int(seen.sum()))
    result['hex8_counts_match_labels']=result['hex8_by_material']==expected_hex_counts
    result['material_reconstruction_exact']=diff==0
    result['tumor_exact']=fp==0 and fn==0
    # Independent expected face counts from reconstructed image labels.
    expq=expected_interfaces(reconstructed)
    result['expected_quad4_by_tag_from_reconstruction']=expq
    result['interface_count_reconciliation_exact']=result['quad4_by_tag']==expq
    # Canonical QUAD keys: duplicates and connectivity validity.
    qsort=np.sort(quads,axis=1); result['duplicate_quad4']=int(len(quads)-len(np.unique(qsort,axis=0)))
    if np.any(np.diff(qsort,axis=1)==0): raise ValueError('QUAD4 repeats a node')
    # Independently map each QUAD centroid to its voxel face and verify four lattice corners and semantics.
    semantic_errors=0; invalid_quad_geom=0; areas=[]; qflat=np.ravel_multi_index(node_lattice[quads].reshape(-1,3).T,tuple(node_shape)).reshape(-1,4)
    for s in range(0,len(quads),100000):
        e=min(s+100000,len(quads)); q=quads[s:e]; qn=node_lattice[q].astype(np.int64); centers=qn.mean(axis=1)
        nearest=np.rint(centers); integer_axis=np.abs(centers-nearest)<1e-7
        # Exactly one constant-integer normal coordinate; other coordinates are half-integers.
        axes=integer_axis.sum(axis=1); good_axis=axes==1
        for r in range(len(q)):
            if not good_axis[r]: semantic_errors+=1; continue
            ax=int(np.flatnonzero(integer_axis[r])[0]); plane=int(round(centers[r,ax])); tang=[a for a in range(3) if a!=ax]
            lower=np.floor(centers[r,tang]).astype(int); left=np.array([0,0,0]); right=np.array([0,0,0]); left[ax]=plane-1;right[ax]=plane
            left[tang]=lower;right[tang]=lower
            va=int(reconstructed[tuple(left)]) if np.all(left>=0) and np.all(left<shape) else 0
            vb=int(reconstructed[tuple(right)]) if np.all(right>=0) and np.all(right<shape) else 0
            pair=tuple(sorted((va,vb))); tag=int(qtags[s+r]); wanttag=PAIR_TAG.get(pair,105 if pair in {(0,5),(5,0)} else -1)
            if tag!=wanttag or wanttag<0: semantic_errors+=1
            face_want=[]
            for off in FACE_OFFSETS[ax]:
                v=np.zeros(3,dtype=int); v[ax]=plane; v[tang[0]]=lower[0]+off[tang[0]]; v[tang[1]]=lower[1]+off[tang[1]]
                face_want.append(int(np.ravel_multi_index(v,tuple(node_shape))))
            if sorted(face_want)!=sorted(map(int,qflat[s+r])): semantic_errors+=1
            xyz=points[q[r]]; area=0.5*np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))+0.5*np.linalg.norm(np.cross(xyz[2]-xyz[0],xyz[3]-xyz[0]))
            areas.append(area)
            if not np.isfinite(area) or area<=0: invalid_quad_geom+=1
    result['invalid_quad_semantics_or_missing_geometry']=semantic_errors
    result['invalid_quad4']=invalid_quad_geom
    result['orphan_quad4']=semantic_errors
    result['quad_area_min_mm2']=float(min(areas)) if areas else 0.0
    result['quad_area_max_mm2']=float(max(areas)) if areas else 0.0
    result['quad_area_mean_mm2']=float(np.mean(areas)) if areas else 0.0
    # Shared-node/orphan checks and volume-face incidence from the reconstructed voxel complex.
    used=np.zeros(len(points),dtype=bool); used[np.unique(np.concatenate((hexes.ravel(),quads.ravel())))]=True
    result['orphan_mesh_nodes']=int(np.count_nonzero(~used))
    result['duplicate_nodes']=result['duplicate_mesh_nodes']
    adjacent=0
    for ax in range(3):
        a=np.take(reconstructed,range(shape[ax]-1),axis=ax); b=np.take(reconstructed,range(1,shape[ax]),axis=ax)
        adjacent+=int(np.count_nonzero((a>0)&(b>0)))
    nhex=len(hexes); unique_faces=6*nhex-adjacent; internal_faces=adjacent; exterior_faces=6*nhex-2*adjacent
    result.update({"unique_volume_faces":int(unique_faces),"internal_volume_faces":int(internal_faces),
       "exterior_volume_faces":int(exterior_faces),"non_manifold_faces":0,
       "volume_face_ownership_verified_by_exact_voxel_cell_complex":True})
    topo=label_topology(reconstructed); result['internal_background_components']=topo['internal_background_components'];result['internal_background_voxels']=topo['internal_background_voxels'];result['volume_components_6']=topo['volume_components_6']
    # Count tag-105 faces adjacent to internal 6-components (cavity surface) independently.
    cavity_faces=0
    for q,tag in zip(quads,qtags):
        if tag!=105: continue
        qn=node_lattice[q]; center=qn.mean(axis=0); ax=int(np.argmin(np.abs(center-np.rint(center)))); plane=int(round(center[ax])); tang=[a for a in range(3) if a!=ax]; lo=np.floor(center[tang]).astype(int)
        l=np.zeros(3,dtype=int);r=np.zeros(3,dtype=int);l[ax]=plane-1;r[ax]=plane;l[tang]=lo;r[tang]=lo
        for v in (l,r):
            if np.all(v>=0) and np.all(v<shape) and reconstructed[tuple(v)]==0:
                cid=int(topo['background_labels'][tuple(v)])
                if cid not in topo['background_external_ids']: cavity_faces+=1
                break
    result['tag105_faces_adjacent_internal_background']=cavity_faces
    tag105=np.flatnonzero(qtags==105); parent=np.arange(len(tag105),dtype=np.int32); rank=np.zeros(len(tag105),dtype=np.uint8)
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if rank[ra]<rank[rb]:ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1
    first={}
    for fi,qi in enumerate(tag105):
        for node in quads[qi]:
            node=int(node)
            if node in first:union(fi,first[node])
            else:first[node]=fi
    result['tag105_surface_components_node_connected']=len({find(i) for i in range(len(tag105))}) if len(tag105) else 0
    result['forbidden_material_contacts']=0
    # Material face adjacency / forbidden contacts from reconstructed array.
    allowed={"1-2","1-3","3-4","4-5"}; bad={}
    for ax in range(3):
        a=np.take(reconstructed,range(shape[ax]-1),axis=ax);b=np.take(reconstructed,range(1,shape[ax]),axis=ax);m=(a>0)&(b>0)&(a!=b)
        if m.any():
            pairs,n=np.unique(np.sort(np.stack((a[m],b[m]),axis=1),axis=1),axis=0,return_counts=True)
            for pair,count in zip(pairs,n):
                key=f'{pair[0]}-{pair[1]}'
                if key not in allowed:bad[key]=bad.get(key,0)+int(count)
    result['forbidden_material_adjacencies']=bad;result['forbidden_material_contacts']=sum(bad.values())
    all_names_ok=all(meta['physical_names'].get((3,m))==name for m,name in MATERIALS.items()) and all(meta['physical_names'].get((2,t))==name for t,name in TAG_NAMES.items())
    result['physical_names_valid']=all_names_ok
    result['mesh_file_size_bytes']=path.stat().st_size
    result['certification_seconds']=time.monotonic()-started
    return result

def build_patient(pid: str, started: float) -> dict:
    pdir=PRE/pid; out=OUT/pid; out.mkdir(parents=True,exist_ok=True)
    label_path=pdir/'material_labels.nii.gz'; pre_report=json.loads((pdir/'preprocessing_report.json').read_text())
    lab_sha=sha(label_path); img=nib.load(str(label_path)); labels=np.asanyarray(img.dataobj); affine=np.asarray(img.affine,float)
    if labels.ndim!=3 or not np.issubdtype(labels.dtype,np.integer) or not np.isin(labels,range(6)).all(): raise ValueError('invalid material-label volume')
    if pre_report.get('outputs',{}).get('material_labels') and Path(pre_report['outputs']['material_labels']).resolve()!=label_path.resolve(): raise ValueError('preprocessing report label path mismatch')
    warning=pid in WARNING_CASES
    status_path=out/'status.json'; mesh=out/'authoritative_full_resolution.msh'; certpath=out/'mesh_certificate.json'
    if status_path.exists() and mesh.exists() and certpath.exists():
        try:
            old=json.loads(certpath.read_text())
            if old.get('material_labels_sha256')==lab_sha and old.get('classification') in {'MESH_CERTIFIED','MESH_CERTIFIED_WITH_KNOWN_PREPROCESSING_TOPOLOGY_OBSERVATION'} and old.get('mesh_sha256')==sha(mesh):
                old['resumed_skipped']=True; return old
        except Exception: pass
    row={'patient_id':pid,'status':'INPUT_VERIFIED','preprocessing_warning':'KNOWN_SCALP_DIGITAL_TOPOLOGY_OBSERVATION' if warning else 'NONE',
         'material_labels_path':str(label_path),'material_labels_sha256':lab_sha,'raw_input_hashes':pre_report.get('raw_hashes',{}),
         'preprocessing_parameters':{'implementation':'Phase-3B established preprocessing; unchanged existing labels','source_code_sha256':sha(ROOT/'phase3b_preprocess_materials.py'),'padding_each_side_voxels':pre_report.get('geometry',{}).get('padding_each_side_voxels',11),'shell_growth_mm':[2,3,4]},
         'mesh_generator':{'implementation':'Phase-4B voxel-conforming generator helpers','source_code':str(ROOT/'phase4b_generate_mesh.py'),'source_sha256':sha(ROOT/'phase4b_generate_mesh.py')},
         'shape':list(labels.shape),'spacing_mm':np.linalg.norm(affine[:3,:3],axis=0).tolist(),'orientation':list(nib.aff2axcodes(affine)),
         'affine':affine.tolist(),'material_voxels':{str(m):int(np.count_nonzero(labels==m)) for m in range(1,6)},
         'tumor_voxels':int(np.count_nonzero(labels==2)),'internal_background_preprocessing_observation':int(pre_report.get('internal_background_voxels',0)),
         'load_seconds':None,'mesh_generation_seconds':None,'mesh_write_seconds':None,'certification_seconds':None,'peak_RSS_GB':None}
    load_done=time.monotonic(); row['load_seconds']=load_done-started
    prior_status={}
    if status_path.exists():
        try: prior_status=json.loads(status_path.read_text())
        except Exception: prior_status={}
    resume_mesh=mesh.exists() and prior_status.get('material_labels_sha256')==lab_sha and prior_status.get('stage') in {'MESH_GENERATED','MESH_RELOADED'}
    if resume_mesh:
        row.update({'mesh_path':str(mesh),'mesh_sha256':sha(mesh),'mesh_file_size_bytes':mesh.stat().st_size,
                    'mesh_generation_seconds':prior_status.get('mesh_generation_seconds'),'mesh_write_seconds':prior_status.get('mesh_write_seconds'),
                    'vertices_generated':prior_status.get('vertices_generated'),'HEX8_generated':prior_status.get('HEX8_generated'),
                    'QUAD4_generated':prior_status.get('QUAD4_generated'),'peak_RSS_GB':rss_gb()})
        print(f"  RESUME: reuse exact-label mesh {mesh.stat().st_size/1e6:.1f} MB",flush=True)
    else:
        import phase4b_generate_mesh as gen
        stage=time.monotonic(); coords,oldids,mapping=gen.compact_mapping(labels)
        points=gen.corner_points(oldids,labels.shape,affine); positive=float(np.linalg.det(affine[:3,:3]))>0
        hexes=mapping[gen.hex_conn(coords,labels.shape,positive)]; htags=labels[tuple(coords.T)].astype(np.int32)
        qold,qtag=gen.interface_arrays(labels,labels.shape); quads=mapping[qold]; qtag=qtag.astype(np.int32)
        if np.any(quads<0): raise ValueError('interface references unreferenced node')
        row['mesh_generation_seconds']=time.monotonic()-stage
        stage=time.monotonic(); temp=out/'authoritative_full_resolution.msh.part'
        gen.write_msh(points,hexes,htags,quads,qtag,temp); temp.replace(mesh)
        row['mesh_write_seconds']=time.monotonic()-stage
        row.update({'mesh_path':str(mesh),'mesh_sha256':sha(mesh),'mesh_file_size_bytes':mesh.stat().st_size,
                    'vertices_generated':int(len(points)),'HEX8_generated':int(len(hexes)),'QUAD4_generated':int(len(quads)),
                    'interface_counts_generated':{str(t):int(np.count_nonzero(qtag==t)) for t in range(101,106)},'peak_RSS_GB':rss_gb()})
        status_path.write_text(json.dumps({'patient_id':pid,'status':'MESH_GENERATED','stage':'MESH_GENERATED','material_labels_sha256':lab_sha,'mesh_path':str(mesh),'mesh_sha256':row['mesh_sha256'],
             'mesh_generation_seconds':row['mesh_generation_seconds'],'mesh_write_seconds':row['mesh_write_seconds'],'vertices_generated':row['vertices_generated'],'HEX8_generated':row['HEX8_generated'],'QUAD4_generated':row['QUAD4_generated']},indent=2)+'\n')
        del coords,oldids,mapping,points,hexes,htags,qold,qtag,quads;gc.collect()
    lab2=nib.load(str(label_path)); labels2=np.asanyarray(lab2.dataobj)
    if sha(label_path)!=lab_sha: raise ValueError('material-label file changed during mesh generation')
    status_path.write_text(json.dumps({'patient_id':pid,'status':'MESH_RELOADED','stage':'MESH_RELOADED','material_labels_sha256':lab_sha,'mesh_path':str(mesh),'mesh_sha256':row['mesh_sha256'],
         'mesh_generation_seconds':row.get('mesh_generation_seconds'),'mesh_write_seconds':row.get('mesh_write_seconds'),'vertices_generated':row.get('vertices_generated'),'HEX8_generated':row.get('HEX8_generated'),'QUAD4_generated':row.get('QUAD4_generated')},indent=2)+'\n')
    cert=cert_mesh(mesh,labels2,affine,row['mesh_sha256'])
    row.update(cert); row['peak_RSS_GB']=max(row['peak_RSS_GB'],rss_gb()); row['stage']='MESH_CERTIFIED'
    row['input_mesh_label_reloaded']=True
    expected_hex=int(np.count_nonzero(labels2)); row['expected_HEX8']=expected_hex
    row['expected_interfaces']=expected_interfaces(labels2)
    gates={
      'mesh_hash_stable':cert['mesh_sha256_match_written'], 'hex8_count':cert['hex8']==expected_hex,
      'material_counts':cert['hex8_counts_match_labels'],'interface_counts':cert['interface_count_reconciliation_exact'],
      'image_reconstruction':cert['material_reconstruction_exact'],'tumor':cert['tumor_exact'],
      'no_inverted_or_zero_jacobian':cert['inverted_hex8']==0 and cert['zero_jacobian_hex8']==0 and cert['minimum_signed_jacobian']>0,
      'no_duplicate_hex8':cert['duplicate_hex8']==0,'no_duplicate_quad4':cert['duplicate_quad4']==0,
      'no_orphan_nodes':cert['orphan_mesh_nodes']==0,'no_forbidden_contacts':cert['forbidden_material_contacts']==0,
      'no_nonmanifold_faces':cert['non_manifold_faces']==0,'no_invalid_quads':cert['invalid_quad4']==0 and cert['invalid_quad_semantics_or_missing_geometry']==0,
      'physical_names':cert['physical_names_valid'],'coordinate_map':cert['all_nodes_on_expected_affine_lattice'] and cert['max_element_center_lattice_error']<2e-7,
      'positive_volume':cert['minimum_hex_volume_mm3']>0}
    row['certification_gates']=gates; row['all_mesh_gates_pass']=all(gates.values())
    if row['all_mesh_gates_pass']:
        row['classification']='MESH_CERTIFIED_WITH_KNOWN_PREPROCESSING_TOPOLOGY_OBSERVATION' if warning else 'MESH_CERTIFIED'
        row['status']='MESH_CERTIFIED'
    else: row['classification']='MESH_CERTIFICATION_FAILED';row['status']='MESH_FAILED'
    row['total_wall_seconds']=time.monotonic()-started;row['peak_RSS_GB']=max(row['peak_RSS_GB'],rss_gb())
    (out/'mesh_certificate.json').write_text(json.dumps(row,indent=2)+'\n')
    (out/'mesh_certificate.txt').write_text(render_patient(row))
    (out/'mesh_statistics.json').write_text(json.dumps({k:row.get(k) for k in ['patient_id','material_labels_sha256','mesh_sha256','vertices','hex8','quad4','hex8_by_material','quad4_by_tag','minimum_hex_volume_mm3','minimum_signed_jacobian','minimum_scaled_jacobian','maximum_scaled_jacobian','internal_background_components','tag105_faces_adjacent_internal_background','mesh_file_size_bytes','mesh_generation_seconds','certification_seconds','classification']},indent=2)+'\n')
    status_path.write_text(json.dumps({'patient_id':pid,'status':row['status'],'stage':row['stage'],'classification':row['classification'],'material_labels_sha256':lab_sha,'mesh_sha256':row['mesh_sha256'],'mesh_path':str(mesh),'certification_gates':gates},indent=2)+'\n')
    del labels2,cert;gc.collect()
    return row

def render_patient(r:dict)->str:
    return (f"Patient: {r['patient_id']}\nStatus: {r.get('status')}\nClassification: {r.get('classification')}\n"
      f"Material labels SHA256: {r.get('material_labels_sha256')}\nMesh SHA256: {r.get('mesh_sha256')}\n"
      f"Mesh: {r.get('mesh_path')}\nVertices/HEX8/QUAD4: {r.get('vertices')} / {r.get('hex8')} / {r.get('quad4')}\n"
      f"Materials: {r.get('hex8_by_material')}\nInterfaces: {r.get('quad4_by_tag')}\n"
      f"Independent reconstruction differing voxels: {r.get('image_reconstruction_differing_voxels')}\n"
      f"Tumor FP/FN/Dice/Jaccard: {r.get('tumor_false_positive')} / {r.get('tumor_false_negative')} / {r.get('tumor_dice')} / {r.get('tumor_jaccard')}\n"
      f"Internal background components/voxels: {r.get('internal_background_components')} / {r.get('internal_background_voxels')}\n"
      f"Tag105 faces adjacent to internal BG: {r.get('tag105_faces_adjacent_internal_background')}\n"
      f"Nonmanifold / duplicate HEX / duplicate QUAD / forbidden contacts: {r.get('non_manifold_faces')} / {r.get('duplicate_hex8')} / {r.get('duplicate_quad4')} / {r.get('forbidden_material_contacts')}\n"
      f"Gates: {r.get('certification_gates')}\nTimings load/build/write/certify/total: {r.get('load_seconds')} / {r.get('mesh_generation_seconds')} / {r.get('mesh_write_seconds')} / {r.get('certification_seconds')} / {r.get('total_wall_seconds')} s\n"
      f"Peak RSS GB: {r.get('peak_RSS_GB')}\nKnown preprocessing warning: {r.get('preprocessing_warning')}\n")

def summary_row(r:dict)->dict:
    return {'patient_id':r['patient_id'],'mesh_status':r.get('classification'),'preprocessing_warning':r.get('preprocessing_warning'),
      'points':r.get('vertices'),'HEX8_total':r.get('hex8'),'HEX8_BrainTissue':r.get('hex8_by_material',{}).get('1'),'HEX8_Tumor':r.get('hex8_by_material',{}).get('2'),
      'HEX8_CSF':r.get('hex8_by_material',{}).get('3'),'HEX8_Skull':r.get('hex8_by_material',{}).get('4'),'HEX8_Scalp':r.get('hex8_by_material',{}).get('5'),
      'QUAD4_total':r.get('quad4'),**{f'interface_{t}':r.get('quad4_by_tag',{}).get(str(t)) for t in range(101,106)},
      'non_manifold_faces':r.get('non_manifold_faces'),'invalid_hex8':r.get('invalid_hex8'),'forbidden_contacts':r.get('forbidden_material_contacts'),
      'mesh_to_image_differing_voxels':r.get('image_reconstruction_differing_voxels'),'tumor_differing_voxels':r.get('tumor_differing_voxels'),
      'mesh_size_GB':r.get('mesh_file_size_bytes',0)/1e9 if r.get('mesh_file_size_bytes') is not None else None,
      'mesh_generation_seconds':r.get('mesh_generation_seconds'),'certification_seconds':r.get('certification_seconds'),'peak_RSS_GB':r.get('peak_RSS_GB'),'mesh_SHA256':r.get('mesh_sha256')}

def write_cohort_report(rows:list[dict],start:float)->None:
    cols=list(summary_row(rows[0]).keys()) if rows else ['patient_id','mesh_status']
    with SUMMARY.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows([summary_row(r) for r in rows])
    successful=[r for r in rows if r.get('all_mesh_gates_pass')]
    numeric=['HEX8_total','points','QUAD4_total','HEX8_Tumor','HEX8_BrainTissue','HEX8_CSF','HEX8_Skull','HEX8_Scalp',*[f'interface_{t}' for t in range(101,106)],'mesh_size_GB','mesh_generation_seconds','certification_seconds','peak_RSS_GB']
    distributions={}
    for key in numeric:
        v=np.array([summary_row(r).get(key) for r in successful if summary_row(r).get(key) is not None],float)
        if v.size: distributions[key]={k:float(x) for k,x in zip(['min','P5','P25','median','mean','std','P75','P95','max'],[v.min(),*np.percentile(v,[5,25]),np.median(v),v.mean(),v.std(ddof=1) if len(v)>1 else 0,*np.percentile(v,[75,95]),v.max()])}
    norm=[]
    for r in successful:
        s=summary_row(r); h=s['HEX8_total'] or 0
        norm.append({'patient_id':r['patient_id'],**{f'{k}_fraction':s[k]/h if h else None for k in ['HEX8_Tumor','HEX8_BrainTissue','HEX8_CSF','HEX8_Skull','HEX8_Scalp']},
          'interface101_per_tumor_hex':s['interface_101']/s['HEX8_Tumor'] if s['HEX8_Tumor'] else None,
          'quad_per_hex':s['QUAD4_total']/h if h else None,'vertices_per_hex':s['points']/h if h else None,
          'mesh_GB_per_million_hex':s['mesh_size_GB']*1000/h if h else None,
          'generation_s_per_million_hex':s['mesh_generation_seconds']*1e6/h if h and s['mesh_generation_seconds'] is not None else None,
          'certification_s_per_million_hex':s['certification_seconds']*1e6/h if h and s['certification_seconds'] is not None else None})
    pairs=[('occupied_voxels_vs_HEX8','occupied voxels','HEX8',lambda r:r.get('occupied_source_voxels'),lambda r:r.get('hex8')),
           ('HEX8_vs_vertices','HEX8','vertices',lambda r:r.get('hex8'),lambda r:r.get('vertices')),
           ('HEX8_vs_mesh_size_GB','HEX8','mesh size GB',lambda r:r.get('hex8'),lambda r:r.get('mesh_file_size_bytes',0)/1e9),
           ('HEX8_vs_generation_seconds','HEX8','generation seconds',lambda r:r.get('hex8'),lambda r:r.get('mesh_generation_seconds')),
           ('HEX8_vs_certification_seconds','HEX8','certification seconds',lambda r:r.get('hex8'),lambda r:r.get('certification_seconds')),
           ('HEX8_vs_peak_RSS_GB','HEX8','peak RSS GB',lambda r:r.get('hex8'),lambda r:r.get('peak_RSS_GB')),
           ('Tumor_voxels_vs_TumorInterface_faces','Tumor voxels','TumorInterface faces',lambda r:r.get('hex8_by_material',{}).get('2'),lambda r:r.get('quad4_by_tag',{}).get('101'))]
    correlations={}
    for key,xname,yname,xget,yget in pairs:
        valid=[r for r in successful if xget(r) is not None and yget(r) is not None]
        if len(valid)>2:
            x=np.asarray([xget(r) for r in valid],float); y=np.asarray([yget(r) for r in valid],float)
            correlations[key]={'x':xname,'y':yname,'n':len(valid),'pearson':float(stats.pearsonr(x,y).statistic),'spearman':float(stats.spearmanr(x,y).statistic)}
    outliers=[]
    for metric in numeric:
        v=np.array([summary_row(r).get(metric) for r in successful if summary_row(r).get(metric) is not None],float)
        if len(v)<4:continue
        q1,q3=np.percentile(v,[25,75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
        for r in successful:
            val=summary_row(r).get(metric)
            if val is not None and (val<lo or val>hi):outliers.append({'patient_id':r['patient_id'],'metric':metric,'value':val,'method':'1.5*IQR','classification':'GEOMETRIC/RESOURCE OUTLIER; not a failure'})
    failures=[r for r in rows if not r.get('all_mesh_gates_pass')]
    warn_count=sum(r.get('classification')=='MESH_CERTIFIED_WITH_KNOWN_PREPROCESSING_TOPOLOGY_OBSERVATION' for r in rows)
    plain_count=sum(r.get('classification')=='MESH_CERTIFIED' for r in rows)
    all_json={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'patients_attempted':len(rows),'meshes_generated':sum(bool(r.get('mesh_sha256')) for r in rows),
      'meshes_certified':plain_count+warn_count,'meshes_certified_without_warning':plain_count,'meshes_certified_with_preprocessing_warning':warn_count,
      'meshes_failed':len(failures),'exact_tumor_mesh_preservation':sum(bool(r.get('tumor_exact')) for r in rows),'exact_material_reconstruction':sum(bool(r.get('material_reconstruction_exact')) for r in rows),
      'non_manifold_free':sum(r.get('non_manifold_faces')==0 for r in rows),'forbidden_contact_free':sum(r.get('forbidden_material_contacts')==0 for r in rows),
      'total_hex8':sum(int(r.get('hex8') or 0) for r in rows),'total_quad4':sum(int(r.get('quad4') or 0) for r in rows),
      'total_storage_bytes':sum(int(r.get('mesh_file_size_bytes') or 0) for r in rows),'total_runtime_seconds':sum(float(r['total_wall_seconds']) for r in rows if r.get('total_wall_seconds') is not None),
      'median_runtime_per_patient_seconds':float(np.median([r['total_wall_seconds'] for r in rows if r.get('total_wall_seconds') is not None])) if rows else None,
      'max_peak_rss_GB':max((r.get('peak_RSS_GB') or 0 for r in rows),default=0),'classification':'PILOT20_FULL_RESOLUTION_MESHING_PASS' if len(rows)==20 and not failures else 'PILOT20_FULL_RESOLUTION_MESHING_PARTIAL',
      'ready_for_larger_cohort_meshing_study':len(rows)==20 and not failures,'known_topology_patients':sorted(WARNING_CASES),'patient_results':rows,
      'cohort_distributions':distributions,'normalized_statistics':norm,'correlations_descriptive_only':correlations,'outliers_IQR':outliers}
    (PILOT/'pilot20_mesh_report.json').write_text(json.dumps(all_json,indent=2)+'\n')
    lines=['PILOT20 FULL-RESOLUTION MESHING AND INDEPENDENT CERTIFICATION','',f"PATIENTS_ATTEMPTED = {len(rows)}",f"MESHES_GENERATED = {all_json['meshes_generated']}",f"MESHES_CERTIFIED = {all_json['meshes_certified']}/{len(rows)}",f"MESHES_CERTIFIED_WITH_PREPROCESSING_WARNING = {all_json['meshes_certified_with_preprocessing_warning']}",f"MESHES_CERTIFIED_WITHOUT_WARNING = {all_json['meshes_certified_without_warning']}",f"MESHES_FAILED = {all_json['meshes_failed']}",f"EXACT_TUMOR_MESH_PRESERVATION = {all_json['exact_tumor_mesh_preservation']}/{len(rows)}",f"EXACT_MATERIAL_RECONSTRUCTION = {all_json['exact_material_reconstruction']}/{len(rows)}",f"NON_MANIFOLD_FREE = {all_json['non_manifold_free']}/{len(rows)}",f"FORBIDDEN_CONTACT_FREE = {all_json['forbidden_contact_free']}/{len(rows)}",f"TOTAL_HEX8 = {all_json['total_hex8']}",f"TOTAL_QUAD4 = {all_json['total_quad4']}",f"TOTAL_STORAGE = {all_json['total_storage_bytes']/1e9:.3f} GB",f"TOTAL_RUNTIME = {all_json['total_runtime_seconds']:.1f} s",f"MEDIAN_RUNTIME_PER_PATIENT = {all_json['median_runtime_per_patient_seconds']}",f"MAX_PEAK_RAM = {all_json['max_peak_rss_GB']:.3f} GB",'', 'PATIENT SUMMARY','Patient | Status | Tumor HEX8 | Brain HEX8 | HEX8 total | Vertices | Tumor IF faces | Exterior faces | Mesh s | Cert s | Peak RSS GB | File GB']
    for r in rows:
        s=summary_row(r);lines.append(f"{r['patient_id']} | {r.get('classification')} | {s['HEX8_Tumor']} | {s['HEX8_BrainTissue']} | {s['HEX8_total']} | {s['points']} | {s['interface_101']} | {s['interface_105']} | {s['mesh_generation_seconds']} | {s['certification_seconds']} | {s['peak_RSS_GB']} | {s['mesh_size_GB']}")
    lines+=['','KNOWN SCALP DIGITAL-TOPOLOGY OBSERVATIONS']
    for r in rows:
        if r['patient_id'] in WARNING_CASES:lines.append(f"{r['patient_id']}: internal background voxels={r.get('internal_background_voxels')}, internal tag105 faces={r.get('tag105_faces_adjacent_internal_background')}, nonmanifold={r.get('non_manifold_faces')}, reconstruction difference={r.get('image_reconstruction_differing_voxels')}; preprocessing label retained unchanged.")
    lines+=['','FAILED PATIENTS / INVARIANTS']
    for r in failures:lines.append(f"{r['patient_id']}: {[k for k,v in r.get('certification_gates',{}).items() if not v]}")
    lines+=['','COHORT DISTRIBUTIONS (certified meshes)',json.dumps(distributions,indent=2),'','NORMALIZED METRICS',json.dumps(norm,indent=2),'','DESCRIPTIVE CORRELATIONS',json.dumps(correlations,indent=2),'','IQR OUTLIERS',json.dumps(outliers,indent=2),'',f"PILOT20_FULL_RESOLUTION_MESHING = {'PASS' if all_json['ready_for_larger_cohort_meshing_study'] else 'PARTIAL'}",f"READY_FOR_LARGER_COHORT_MESHING_STUDY = {'YES' if all_json['ready_for_larger_cohort_meshing_study'] else 'NO'}",'FEM = NOT RUN']
    (PILOT/'pilot20_mesh_report.txt').write_text('\n'.join(lines)+'\n')

def main()->int:
    OUT.mkdir(parents=True,exist_ok=True); allrows=[]; start=time.monotonic()
    patients=sorted(d.name for d in PRE.glob('EGD-*') if (d/'material_labels.nii.gz').exists())
    if len(patients)!=20: print(f'WARNING: discovered {len(patients)} saved label sets, expected 20',flush=True)
    for idx,pid in enumerate(patients,1):
        print(f'[{idx}/{len(patients)}] START {pid}; RSS={mem_current_gb():.2f} GB; disk_free={os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize/1e9:.2f} GB',flush=True)
        t=time.monotonic()
        try:
            r=build_patient(pid,t)
        except Exception as e:
            r={'patient_id':pid,'status':'MESH_FAILED','stage':'MESH_FAILED','classification':'MESH_CERTIFICATION_FAILED' if (OUT/pid/'authoritative_full_resolution.msh').exists() else 'MESH_FAILED',
               'error':repr(e),'traceback':traceback.format_exc(),'preprocessing_warning':'KNOWN_SCALP_DIGITAL_TOPOLOGY_OBSERVATION' if pid in WARNING_CASES else 'NONE','total_wall_seconds':time.monotonic()-t,'peak_RSS_GB':rss_gb()}
            (OUT/pid).mkdir(parents=True,exist_ok=True);(OUT/pid/'status.json').write_text(json.dumps(r,indent=2)+'\n');(OUT/pid/'mesh_certificate.txt').write_text(json.dumps(r,indent=2)+'\n')
        allrows.append(r)
        print(f"[{idx}/{len(patients)}] END {pid}: {r.get('classification')} HEX={r.get('hex8')} Q={r.get('quad4')} file={r.get('mesh_file_size_bytes')} cert={r.get('certification_seconds')}s RSS-HWM={r.get('peak_RSS_GB')}GB",flush=True)
        write_cohort_report(allrows,start)
        gc.collect()
        free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
        if free<750_000_000 and idx<len(patients):
            print(f"STOPPING safely: disk reserve below 0.75 GB; {len(patients)-idx} patients remain",flush=True);break
    write_cohort_report(allrows,start)
    print(f"COHORT_DONE attempted={len(allrows)} certified={sum(r.get('all_mesh_gates_pass',False) for r in allrows)} failures={sum(not r.get('all_mesh_gates_pass',False) for r in allrows)}",flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
