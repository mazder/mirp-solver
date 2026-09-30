#!/usr/bin/env python3
"""Independent, read-only certification of the authoritative Gmsh 2.2 mesh.

This intentionally does not import the Phase 4 mesh generator. It parses MSH
2.2 itself, maps element centroids through the independently loaded NIfTI
affine, and derives topology/geometry from mesh connectivity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "EGD-0001/MR_EGD-0001"
MESH = DATASET / "meshing/phase4b_mesh/head_tumor_5region.msh"
LABELS = DATASET / "preprocessing/phase3b_results/material_labels.nii.gz"
TUMOR_SOURCE = DATASET / "5_MASK/NIFTI/MASK.nii.gz"
P3_REPORT = DATASET / "preprocessing/phase3b_results/phase3b_preprocess_report.json"
P4_REPORT = DATASET / "meshing/phase4b_mesh/phase4b_mesh_generation_report.json"
OUT = ROOT / "mesh_validation/authoritative_mesh_certification"
EXPECTED_MESH_SHA = "b66dd853c27c465636fd518c6e03b1540d1538681558778f722e6c2e613ddd85"
EXPECTED = {"hex": {1: 1968564, 2: 82652, 3: 162698, 4: 280585, 5: 383732},
            "quad": {101: 21094, 102: 131310, 103: 137498, 104: 147048, 105: 159038},
            "vertices": 2958278}
NAMES = {1:"BrainTissue", 2:"Tumor", 3:"CSF", 4:"Skull", 5:"Scalp",
         101:"TumorInterface", 102:"BrainCSFInterface", 103:"CSFSkullInterface",
         104:"SkullScalpInterface", 105:"ScalpExterior"}
PAIR_TAG = {(1,2):101, (1,3):102, (3,4):103, (4,5):104}
HEX_FACES = ((0,3,7,4),(1,2,6,5),(0,1,5,4),(3,2,6,7),(0,1,2,3),(4,5,6,7))
T0 = time.perf_counter()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def stage(s: str) -> None:
    print(f"[{time.perf_counter()-T0:9.1f}s] {s}; RSS={rss_bytes()/1e9:.3f} GB", flush=True)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def parse_msh(path: Path):
    """Read MSH 2.2 ASCII nodes + type 3 QUAD4/type 5 HEX8, preserving extras."""
    physical, version, nodes, hexes, htags, quads, qtags, unexpected = {}, None, None, [], [], [], [], Counter()
    with path.open("r", buffering=4*1024*1024) as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            if line == "$MeshFormat":
                version = f.readline().split()[0]
                f.readline()
            elif line == "$PhysicalNames":
                n = int(f.readline())
                for _ in range(n):
                    p = f.readline().split(maxsplit=2); physical[(int(p[0]),int(p[1]))] = p[2].strip().strip('"')
                f.readline()
            elif line == "$Nodes":
                n = int(f.readline()); nodes = np.empty((n,3), dtype=np.float64)
                node_ids = np.empty(n, dtype=np.int64)
                for i in range(n):
                    p = f.readline().split(); node_ids[i] = int(p[0]); nodes[i] = (float(p[1]),float(p[2]),float(p[3]))
                if not np.array_equal(node_ids, np.arange(1,n+1)):
                    raise ValueError("Gmsh node identifiers are not consecutive 1-based; remap required")
                f.readline()
            elif line == "$Elements":
                n = int(f.readline())
                for ix in range(n):
                    p = list(map(int, f.readline().split()))
                    etype, ntags = p[1], p[2]
                    tags = p[3:3+ntags]; conn = p[3+ntags:]
                    if etype == 5 and len(conn) == 8:
                        hexes.append(conn); htags.append(tags[0] if tags else 0)
                    elif etype == 3 and len(conn) == 4:
                        quads.append(conn); qtags.append(tags[0] if tags else 0)
                    else:
                        unexpected[etype] += 1
                    if ix and ix % 1_000_000 == 0: stage(f"parsed {ix:,}/{n:,} mesh elements")
                f.readline()
    return (version, physical, nodes, np.asarray(hexes,dtype=np.int32)-1,
            np.asarray(htags,dtype=np.int16), np.asarray(quads,dtype=np.int32)-1,
            np.asarray(qtags,dtype=np.int16), dict(unexpected))


def face_keys(faces: np.ndarray) -> np.ndarray:
    # Structured canonical key enables compact sorting/search without Python tuples.
    a = np.sort(faces, axis=1).astype(np.int32, copy=False)
    dt = np.dtype([("a","<i4"),("b","<i4"),("c","<i4"),("d","<i4")])
    return a.view(dt).reshape(-1)


def qstats(a):
    return {"min": float(np.min(a)), "max": float(np.max(a)), "mean": float(np.mean(a))}


def image_components(labels: np.ndarray):
    st = ndimage.generate_binary_structure(3,1)
    out = {}
    for m in range(1,6):
        lab,n = ndimage.label(labels == m, structure=st)
        counts = np.bincount(lab.ravel())[1:]
        out[str(m)] = {"components":int(n),"largest":int(counts.max()) if len(counts) else 0,
                       "small_component_sizes":sorted(map(int,counts[counts < 100]))}
        del lab, counts
    bg_lab,bg_n = ndimage.label(labels == 0, structure=st)
    flat = bg_lab.ravel(); boundary = np.zeros(bg_n+1,dtype=bool)
    for ax in range(3):
        for sl in (tuple(slice(None) if d != ax else 0 for d in range(3)),
                   tuple(slice(None) if d != ax else -1 for d in range(3))):
            boundary[np.unique(bg_lab[sl])] = True
    sizes = np.bincount(flat, minlength=bg_n+1)
    internal_ids = np.flatnonzero(~boundary); internal_ids = internal_ids[internal_ids != 0]
    return out, {"background_components_total":int(bg_n),"internal_components":int(len(internal_ids)),
                 "internal_voxels":int(sizes[internal_ids].sum()) if len(internal_ids) else 0,
                 "background_boundary_component_sizes":sorted(map(int,sizes[np.flatnonzero(boundary)[1:]]))[:20]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh_sha = sha(MESH); label_sha = sha(LABELS); tumor_sha = sha(TUMOR_SOURCE)
    p3 = json.loads(P3_REPORT.read_text()); p4 = json.loads(P4_REPORT.read_text())
    if mesh_sha != EXPECTED_MESH_SHA:
        raise RuntimeError(f"AUTHORITATIVE MESH CHECKSUM MISMATCH: {mesh_sha}")
    stage("integrity hashes verified; parsing mesh")
    version, physical, xyz, hexes, hm, quads, qm, unexpected = parse_msh(MESH)
    stage("mesh parsed; loading authoritative labels and tumor source")
    img = nib.load(str(LABELS)); labels = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine,dtype=np.float64); inv_aff = np.linalg.inv(affine)
    tumor_img = nib.load(str(TUMOR_SOURCE)); tumor_source = np.asanyarray(tumor_img.dataobj) > 0
    shape = labels.shape; voxel_volume = abs(float(np.linalg.det(affine[:3,:3])))

    # Image-space independent counts, physical summaries and connected components.
    label_counts = {str(m):int(np.count_nonzero(labels==m)) for m in range(6)}
    invalid_label_values = sorted(set(map(int,np.unique(labels))) - set(range(6)))
    img_comps,bg_top = image_components(labels)
    occ = labels > 0
    pad = int(p3["padding_voxels_each_side"])
    embedded = np.zeros(shape,dtype=bool)
    if tuple(np.array(tumor_source.shape)+2*pad) != tuple(shape):
        raise RuntimeError("Original tumor shape plus documented padding does not match authoritative label grid")
    embedded[pad:pad+tumor_source.shape[0],pad:pad+tumor_source.shape[1],pad:pad+tumor_source.shape[2]] = tumor_source
    tumor_img_vs_labels = {"source_voxels":int(tumor_source.sum()),"label_tumor_voxels":int(np.count_nonzero(labels==2)),
                           "missing":int(np.count_nonzero(embedded & (labels!=2))),
                           "extra":int(np.count_nonzero((labels==2) & ~embedded))}

    # Basic mesh inventory and connectivity validity.
    hcnt = {int(k):int(v) for k,v in zip(*np.unique(hm,return_counts=True))}
    qcnt = {int(k):int(v) for k,v in zip(*np.unique(qm,return_counts=True))}
    referenced = np.unique(np.concatenate((hexes.ravel(),quads.ravel())))
    repeated_nodes = int(np.count_nonzero(np.any(np.diff(np.sort(hexes,axis=1),axis=1)==0,axis=1)))
    # Hex duplicate detection uses canonical node tuples.
    hs = np.sort(hexes,axis=1); hdt=np.dtype([(f"n{i}","<i4") for i in range(8)])
    hkeys=hs.view(hdt).reshape(-1); horder=np.argsort(hkeys,kind="quicksort")
    duplicate_hexes=int(np.count_nonzero(hkeys[horder][1:]==hkeys[horder][:-1]))
    del hs,hkeys,horder
    coord_keys=np.ascontiguousarray(xyz).view(np.dtype((np.void,xyz.dtype.itemsize*3))).ravel()
    unique_coordinate_count=int(np.unique(coord_keys).size); del coord_keys
    stage("basic inventory complete; constructing canonical HEX8 face ownership")

    # Construct six faces/HEX, then lexicographically group all canonical faces.
    nf=len(hexes)*6
    faces=np.empty((nf,4),dtype=np.int32)
    owners=np.repeat(np.arange(len(hexes),dtype=np.int32),6)
    for fi,loc in enumerate(HEX_FACES): faces[fi::6]=hexes[:,loc]
    faces.sort(axis=1)
    fdt=np.dtype([("a","<i4"),("b","<i4"),("c","<i4"),("d","<i4")])
    fkeys=faces.view(fdt).reshape(-1)
    order=np.argsort(fkeys,kind="quicksort")
    sk=fkeys[order]; starts=np.r_[0,np.flatnonzero(sk[1:]!=sk[:-1])+1].astype(np.int64)
    counts=np.diff(np.r_[starts,len(sk)]).astype(np.uint8)
    unique_keys=sk[starts]
    unique_owner=owners[order[starts]]
    unique_mat=hm[unique_owner]
    second_owner=np.full(len(starts),-1,dtype=np.int32)
    paired=counts==2
    second_owner[paired]=owners[order[starts[paired]+1]]
    nonmanifold=int(np.count_nonzero(counts>2)); zero_owner=0
    internal=int(np.count_nonzero(counts==2)); exterior=int(np.count_nonzero(counts==1))
    stage(f"face grouping complete: {len(starts):,} unique volume faces")

    # Match every serialized QUAD exactly to one canonical volume face.
    qkeys=face_keys(quads); qsort=np.sort(qkeys,kind="quicksort")
    qdup=int(np.count_nonzero(qsort[1:]==qsort[:-1]))
    qidx=np.searchsorted(unique_keys,qkeys)
    inrange=qidx<len(unique_keys); matched=np.zeros(len(quads),dtype=bool)
    matched[inrange]=unique_keys[qidx[inrange]]==qkeys[inrange]
    qowners=np.full((len(quads),2),-1,dtype=np.int32)
    qowners[matched,0]=unique_owner[qidx[matched]]
    qowners[matched,1]=second_owner[qidx[matched]]
    orphan=int(np.count_nonzero(~matched))
    q_bad_sem=0
    for i,tag in enumerate(qm):
        if not matched[i]: continue
        a=int(hm[qowners[i,0]]); b=int(hm[qowners[i,1]]) if qowners[i,1]>=0 else 0
        if tag in (101,102,103,104):
            pair=tuple(sorted((a,b)))
            if b == 0 or PAIR_TAG.get(pair) != int(tag): q_bad_sem += 1
        elif tag == 105:
            if b != 0 or a != 5: q_bad_sem += 1
        else: q_bad_sem += 1
    # Exact expected surface set: every cross-material and every exterior face has one QUAD.
    needed=np.zeros(len(starts),dtype=bool)
    needed[paired]=unique_mat[paired] != hm[second_owner[paired]]
    needed[counts==1]=unique_mat[counts==1] == 5
    expected_face_keys=unique_keys[needed]
    actual_needed=qkeys[matched]
    missing_quads=int(len(expected_face_keys)-np.count_nonzero(np.isin(expected_face_keys,actual_needed)))
    unexpected_quads=int(len(quads)-orphan-qdup-(len(expected_face_keys)-missing_quads))
    # Material adjacency face counts.
    cross=paired & (unique_mat != hm[np.maximum(second_owner,0)])
    adj=Counter()
    for a,b in zip(unique_mat[cross],hm[second_owner[cross]]): adj[tuple(sorted((int(a),int(b))))]+=1
    same_internal=int(np.count_nonzero(paired & ~cross))
    forbidden={"1-4","1-5","2-3","2-4","2-5","3-5"}
    adj_text={f"{a}-{b}":v for (a,b),v in sorted(adj.items())}
    forbidden_count=sum(v for k,v in adj_text.items() if k in forbidden)
    exterior_by_mat={str(m):int(np.count_nonzero((counts==1)&(unique_mat==m))) for m in range(1,6)}
    stage("interface ownership and material adjacency derived")

    # HEX geometry, every element. Local order maps index axes to x/y/z.
    hp=xyz[hexes]
    edge_vectors=np.stack((hp[:,1]-hp[:,0],hp[:,3]-hp[:,0],hp[:,4]-hp[:,0]),axis=2)
    signed_jac=np.linalg.det(edge_vectors)
    vols=signed_jac.copy()
    # For this affine voxel lattice the Jacobian is constant over each HEX.
    eids=((0,1,3),(1,2,0),(2,3,1),(3,0,2),(4,5,7),(5,6,4),(6,7,5),(7,4,6))
    sj=[]
    local_bits=((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))
    bit_to_corner={b:i for i,b in enumerate(local_bits)}
    for corner in range(8):
        # construct oriented x/y/z directions at each reference corner
        ijk=local_bits[corner]
        vec=[]
        for axis in range(3):
            nxt=list(ijk); nxt[axis]=1-nxt[axis]
            target=bit_to_corner[tuple(nxt)]
            v=hp[:,target]-hp[:,corner]
            if ijk[axis]==1: v=-v
            vec.append(v)
        det=np.einsum("ij,ij->i",vec[0],np.cross(vec[1],vec[2]))
        denom=np.linalg.norm(vec[0],axis=1)*np.linalg.norm(vec[1],axis=1)*np.linalg.norm(vec[2],axis=1)
        sj.append(det/denom)
    sj=np.stack(sj,axis=1)
    all_edges=np.stack([hp[:,b]-hp[:,a] for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))],axis=1)
    edge_lengths=np.linalg.norm(all_edges,axis=2)
    el_min=edge_lengths.min(axis=1); el_max=edge_lengths.max(axis=1)
    aspect=el_max/el_min
    scale=max(float(np.max(np.abs(vols))),1.0); near_tol=1e-12*scale
    geom={"volume":qstats(vols),"signed_jacobian_center":qstats(signed_jac),"scaled_jacobian_vertex":qstats(sj),
          "edge_length":qstats(edge_lengths),"maximum_aspect_ratio":float(aspect.max()),
          "negative_jacobian_elements":int(np.count_nonzero(signed_jac<0)),
          "zero_jacobian_elements":int(np.count_nonzero(signed_jac==0)),
          "near_degenerate_elements":int(np.count_nonzero(np.abs(signed_jac)<=near_tol)),
          "zero_volume_elements":int(np.count_nonzero(vols==0)),"inverted_elements":int(np.count_nonzero(signed_jac<0)),
          "jacobian_convention":"det([dx/dxi, dx/deta, dx/dzeta]) using HEX8 local axes and reference coordinates [0,1]^3; evaluated at center (constant affine voxel map); scaled Jacobian evaluated at all 8 corners"}
    # QUAD physical area and normal-direction bins from serialized order.
    qp=xyz[quads]; qcross=np.cross(qp[:,1]-qp[:,0],qp[:,3]-qp[:,0]); qnorm=np.linalg.norm(qcross,axis=1)
    # The quadrilateral is planar; two-triangle area avoids orientation dependence.
    qarea=0.5*(np.linalg.norm(np.cross(qp[:,1]-qp[:,0],qp[:,2]-qp[:,0]),axis=1)+
               np.linalg.norm(np.cross(qp[:,2]-qp[:,0],qp[:,3]-qp[:,0]),axis=1))
    quad_area_summary={}
    for tag in (101,102,103,104,105):
        sel=qm==tag; norms=np.abs(qcross[sel]); bins=np.argmax(norms,axis=1) if np.any(sel) else np.empty(0,int)
        quad_area_summary[str(tag)]={"count":int(sel.sum()),"area_mm2":float(qarea[sel].sum()),
             "min_area_mm2":float(qarea[sel].min()) if np.any(sel) else None,
             "max_area_mm2":float(qarea[sel].max()) if np.any(sel) else None,
             "mean_area_mm2":float(qarea[sel].mean()) if np.any(sel) else None,
             "normal_axis_counts_abs":{a:int(np.count_nonzero(bins==i)) for i,a in enumerate("xyz")}}
    qplanar=np.abs(np.einsum("ij,ij->i",qp[:,2]-qp[:,0],qcross))/np.maximum(qnorm,1e-300)
    qrepeat=np.any(np.diff(np.sort(quads,axis=1),axis=1)==0,axis=1)
    invalid_quad=int(np.count_nonzero((qarea<=0)|~np.isfinite(qarea)|(qplanar>1e-10)|qrepeat))
    stage("geometry quality evaluated for all elements")

    # Map element centroids to the authoritative label voxel via inverse NIfTI affine.
    centers=hp.mean(axis=1); ijk_float=(np.c_[centers,np.ones(len(centers))] @ inv_aff.T)[:,:3]
    ijk=np.rint(ijk_float).astype(np.int32); coord_error=float(np.max(np.abs(ijk_float-ijk)))
    in_grid=np.all((ijk>=0)&(ijk<np.asarray(shape)),axis=1)
    mesh_labels=np.zeros(len(hexes),dtype=np.int16)
    mesh_labels[in_grid]=labels[tuple(ijk[in_grid].T)]
    coordinate_map_bad=int(np.count_nonzero(~in_grid | (mesh_labels!=hm)))
    linear=np.ravel_multi_index(ijk[in_grid].T,shape)
    unique_cells=np.unique(linear).size
    reconstructed=np.zeros(shape,dtype=np.uint8)
    reconstructed[tuple(ijk[in_grid].T)]=hm[in_grid].astype(np.uint8)
    diff=reconstructed!=labels
    # Physical-coordinate test: all 8 mesh corners vs affine(index + corner - 0.5).
    coords_expected=np.empty_like(hp)
    for c in range(8):
        off=np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))[c],float)-0.5
        coords_expected[:,c]=(np.c_[ijk, np.ones(len(ijk))] @ affine.T)[:,:3] + affine[:3,:3]@off
    coord_discrepancy=np.linalg.norm(hp-coords_expected,axis=2)
    max_coord_discrepancy=float(np.max(coord_discrepancy))
    # Material summaries in image and mesh, with world-space centers/bounds.
    material_summary={}
    for m in range(1,6):
        ij=np.argwhere(labels==m); points=(np.c_[ij,np.ones(len(ij))] @ affine.T)[:,:3]
        msel=hm==m; mcenters=centers[msel]
        material_summary[str(m)]={"name":NAMES[m],"source_voxels":int(len(ij)),"mesh_hexes":int(msel.sum()),
             "count_difference":int(msel.sum()-len(ij)),"source_volume_mm3":float(len(ij)*voxel_volume),
             "mesh_volume_mm3":float(vols[msel].sum()),"volume_abs_difference_mm3":float(abs(vols[msel].sum()-len(ij)*voxel_volume)),
             "volume_relative_difference":float(abs(vols[msel].sum()-len(ij)*voxel_volume)/max(len(ij)*voxel_volume,1)),
             "source_centroid_world":points.mean(axis=0).tolist(),"mesh_centroid_world":mcenters.mean(axis=0).tolist(),
             "centroid_discrepancy_mm":float(np.linalg.norm(points.mean(axis=0)-mcenters.mean(axis=0))),
             "source_bbox_world":{"min":points.min(axis=0).tolist(),"max":points.max(axis=0).tolist()},
             "mesh_bbox_world":{"min":xyz[hexes[msel]].reshape(-1,3).min(axis=0).tolist(),"max":xyz[hexes[msel]].reshape(-1,3).max(axis=0).tolist()}}
        del ij,points
    # Tumor source (un-padded) geometric exactness and spatial metrics.
    tmask=labels==2; tco=np.argwhere(tmask); tumor_world=(np.c_[tco,np.ones(len(tco))] @ affine.T)[:,:3]
    original_world=(np.argwhere(tumor_source)+np.array([pad,pad,pad])); original_world=(np.c_[original_world,np.ones(len(original_world))] @ affine.T)[:,:3]
    tumor_voxel_exact=bool(np.array_equal(tmask,embedded))
    tumor={"source_voxels":int(tumor_source.sum()),"mesh_tumor_hexes":int(np.count_nonzero(hm==2)),
           "false_negative":int(np.count_nonzero(embedded & ~tmask)),"false_positive":int(np.count_nonzero(tmask & ~embedded)),
           "differing_voxels":int(np.count_nonzero(tmask!=embedded)),"dice":float(2*np.count_nonzero(tmask&embedded)/(tmask.sum()+embedded.sum())),
           "jaccard":float(np.count_nonzero(tmask&embedded)/np.count_nonzero(tmask|embedded)),
           "source_centroid_world":original_world.mean(axis=0).tolist(),"label_centroid_world":tumor_world.mean(axis=0).tolist(),
           "centroid_difference_mm":float(np.linalg.norm(original_world.mean(axis=0)-tumor_world.mean(axis=0))),
           "source_bbox_index":{"min":np.argwhere(tumor_source).min(axis=0).tolist(),"max":np.argwhere(tumor_source).max(axis=0).tolist()},
           "label_bbox_index":{"min":tco.min(axis=0).tolist(),"max":tco.max(axis=0).tolist()},"exact":tumor_voxel_exact}
    stage("image/mesh voxel mapping and tumor comparison complete")

    # Connected components from independently derived mesh-face adjacency.
    row=unique_owner[paired].astype(np.int32); col=second_owner[paired].astype(np.int32)
    graph=sparse.coo_matrix((np.ones(2*len(row),dtype=np.uint8),(np.r_[row,col],np.r_[col,row])),shape=(len(hexes),len(hexes))).tocsr()
    ncomp,comp=connected_components(graph,directed=False,return_labels=True)
    mesh_components={"all":{"components":int(ncomp),"largest":int(np.bincount(comp).max())}}
    del graph,comp
    for m in range(1,6):
        ids=np.flatnonzero(hm==m)
        local=np.full(len(hexes),-1,dtype=np.int32); local[ids]=np.arange(len(ids),dtype=np.int32)
        use=(hm[row]==m) & (hm[col]==m)
        rr=local[row[use]]; cc=local[col[use]]
        mg=sparse.coo_matrix((np.ones(2*len(rr),dtype=np.uint8),(np.r_[rr,cc],np.r_[cc,rr])),shape=(len(ids),len(ids))).tocsr()
        nm,mlab=connected_components(mg,directed=False,return_labels=True)
        labs=np.bincount(mlab)
        mesh_components[str(m)]={"components":int(len(labs)),"largest":int(labs.max()) if len(labs) else 0,
                                 "small_component_sizes":sorted(map(int,labs[labs<100]))}
        del local,mg,mlab,labs,ids,rr,cc

    # Landmark checks distributed deterministically through sorted occupied voxel flat indices.
    occupied_flat=np.flatnonzero(occ); positions=np.linspace(0,len(occupied_flat)-1,17,dtype=np.int64)
    landmark_records=[]
    for flat in occupied_flat[positions]:
        idx=np.array(np.unravel_index(int(flat),shape)); cphys=(affine @ np.r_[idx,1.])[:3]
        hit=np.flatnonzero(np.all(ijk==idx,axis=1))
        if len(hit):
            e=int(hit[0]); expected_corners=(np.c_[idx+np.asarray(local_bits)-0.5,np.ones(8)] @ affine.T)[:,:3]
            actual_corners=hp[e].tolist()
            landmark_records.append({"voxel_index":idx.tolist(),"physical_center":cphys.tolist(),"element_count_at_voxel":int(len(hit)),
                "mesh_attribute":int(hm[e]),"mesh_element_index_zero_based":e,"mesh_node_ids_zero_based":hexes[e].tolist(),
                "expected_physical_corners":expected_corners.tolist(),"mesh_physical_corners":actual_corners,
                "max_corner_discrepancy_mm":float(np.linalg.norm(hp[e]-expected_corners,axis=1).max())})
        else:
            landmark_records.append({"voxel_index":idx.tolist(),"physical_center":cphys.tolist(),"element_count_at_voxel":0,"mesh_attribute":None})

    # Mesh/image physical domains from voxel corners.
    occ_idx=np.argwhere(occ); imin=occ_idx.min(axis=0); imax=occ_idx.max(axis=0)
    corner_idx=np.array([[i,j,k] for i in (imin[0]-.5,imax[0]+.5) for j in (imin[1]-.5,imax[1]+.5) for k in (imin[2]-.5,imax[2]+.5)])
    image_bbox=(np.c_[corner_idx,np.ones(8)] @ affine.T)[:,:3]
    mesh_bbox={"min":xyz.min(axis=0).tolist(),"max":xyz.max(axis=0).tolist()}
    surface_components={}
    # QUAD components are computed by shared-node graph separately per tag.
    for tag in range(101,106):
        qi=np.flatnonzero(qm==tag); conn=quads[qi]
        sel=qm==tag; normals=qcross[sel]
        if len(normals):
            axes=np.argmax(np.abs(normals),axis=1); signs=np.sign(normals[np.arange(len(normals)),axes]).astype(int)
            direction_counts={f"{axis}{sign:+d}":int(np.count_nonzero((axes==ai)&(signs==sign)))
                              for ai,axis in enumerate("xyz") for sign in (-1,1)}
            surfpts=xyz[conn].reshape(-1,3)
            surf_bbox={"min":surfpts.min(axis=0).tolist(),"max":surfpts.max(axis=0).tolist()}
        else:
            direction_counts={f"{axis}{sign:+d}":0 for axis in "xyz" for sign in (-1,1)}; surf_bbox=None
        if len(qi):
            # Sparse node-to-quad incidence then adjacency via incidence product.
            rr=np.repeat(np.arange(len(qi)),4); cc=conn.ravel()
            inc=sparse.coo_matrix((np.ones(len(rr),dtype=np.uint8),(rr,cc)),shape=(len(qi),len(xyz))).tocsr()
            qg=(inc @ inc.T).tocsr(); qg.setdiag(0); qg.eliminate_zeros()
            nc,cl=connected_components(qg,directed=False,return_labels=True)
            surface_components[str(tag)]={"components":int(nc),"largest_faces":int(np.bincount(cl).max())}
        else: surface_components[str(tag)]={"components":0,"largest_faces":0}
        surface_components[str(tag)]["normal_directions_from_serialized_winding"]=direction_counts
        surface_components[str(tag)]["bbox_world"]=surf_bbox
    stage("connected components and surface characterization complete")

    # Landmark physical coordinates establish half-voxel and affine convention.
    integrity = mesh_sha==EXPECTED_MESH_SHA
    gates={
      "mesh_sha256_match":integrity,
      "expected_element_counts":hcnt==EXPECTED["hex"] and qcnt==EXPECTED["quad"] and len(xyz)==EXPECTED["vertices"],
      "valid_material_ids":set(hcnt)==set(range(1,6)) and not invalid_label_values,
      "valid_interface_ids":set(qcnt)==set(range(101,106)),
      "physical_coordinate_mapping":coordinate_map_bad==0 and unique_cells==len(hexes) and max_coord_discrepancy<1e-9 and np.allclose(mesh_bbox["min"],image_bbox.min(axis=0),atol=1e-9) and np.allclose(mesh_bbox["max"],image_bbox.max(axis=0),atol=1e-9),
      "inverted_hex8_zero":geom["inverted_elements"]==0,
      "degenerate_hex8_zero":geom["zero_volume_elements"]==0 and geom["near_degenerate_elements"]==0,
      "nonmanifold_faces_zero":nonmanifold==0,
      "cracks_or_nonconforming_faces_zero":unique_coordinate_count==len(xyz),
      "forbidden_material_adjacencies_zero":forbidden_count==0,
      "unintended_internal_background_zero":bg_top["internal_components"]==0 and bg_top["internal_voxels"]==0,
      "interface_reconciliation_exact":orphan==0 and qdup==0 and q_bad_sem==0 and missing_quads==0 and unexpected_quads==0,
      "orphan_quad4_zero":orphan==0,
      "duplicate_quad4_zero":qdup==0,
      "invalid_quad4_zero":invalid_quad==0,
      "image_to_mesh_material_mapping_exact":coordinate_map_bad==0 and unique_cells==len(hexes) and not np.any(diff),
      "tumor_preservation_exact":tumor["exact"] and tumor["false_negative"]==0 and tumor["false_positive"]==0,
      "mesh_to_image_reconstruction_exact":not np.any(diff),
      "material_volume_reconciliation_pass":all(material_summary[str(m)]["count_difference"]==0 and material_summary[str(m)]["volume_relative_difference"]<1e-12 for m in range(1,6)),
      "connectivity_valid":int(referenced.min())==0 and int(referenced.max())<len(xyz) and repeated_nodes==0 and duplicate_hexes==0 and len(referenced)==len(xyz),
      "image_mesh_component_structure_matches":all(mesh_components[str(m)]["components"]==img_comps[str(m)]["components"] for m in range(1,6)),
      "only_scalp_contacts_exterior":all(exterior_by_mat[str(m)]==0 for m in range(1,5)),
      "all_boundary_faces_have_quad":missing_quads==0,
      "all_expected_material_counts_match_labels":all(label_counts[str(m)]==EXPECTED["hex"][m] for m in range(1,6)),
      "tumor_source_padding_mapping_exact":tumor_img_vs_labels["missing"]==0 and tumor_img_vs_labels["extra"]==0,
    }
    certified=all(gates.values()) and not unexpected
    # emit evidence tables
    write_csv(OUT/"material_summary.csv",["attribute","material","source_voxels","mesh_hexes","count_difference","source_volume_mm3","mesh_volume_mm3","relative_volume_difference","components_mesh","components_image"],
       [[m,NAMES[m],material_summary[str(m)]["source_voxels"],material_summary[str(m)]["mesh_hexes"],material_summary[str(m)]["count_difference"],material_summary[str(m)]["source_volume_mm3"],material_summary[str(m)]["mesh_volume_mm3"],material_summary[str(m)]["volume_relative_difference"],mesh_components[str(m)]["components"],img_comps[str(m)]["components"]] for m in range(1,6)])
    write_csv(OUT/"material_adjacency.csv",["material_a","material_b","shared_internal_hex_faces","allowed"],
       [[a,b,adj.get((a,b),0),((a,b) in PAIR_TAG)] for a in range(1,6) for b in range(a+1,6)])
    write_csv(OUT/"interface_summary.csv",["tag","name","faces","area_mm2","min_area_mm2","max_area_mm2","mean_area_mm2","components","normal_x_abs","normal_y_abs","normal_z_abs"],
       [[t,NAMES[t],quad_area_summary[str(t)]["count"],quad_area_summary[str(t)]["area_mm2"],quad_area_summary[str(t)]["min_area_mm2"],quad_area_summary[str(t)]["max_area_mm2"],quad_area_summary[str(t)]["mean_area_mm2"],surface_components[str(t)]["components"],*[quad_area_summary[str(t)]["normal_axis_counts_abs"][a] for a in "xyz"]] for t in range(101,106)])
    write_csv(OUT/"mesh_quality_summary.csv",["metric","value"],[[k,json.dumps(v,sort_keys=True)] for k,v in geom.items()])

    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"classification":"AUTHORITATIVE_PATIENT_MESH_GEOMETRICALLY_AND_TOPOLOGICALLY_CERTIFIED" if certified else "MESH_CERTIFICATION_FAILED",
      "mesh_certified":bool(certified),"meshing_mvp_complete":bool(certified),"mesh_read_only":True,
      "artifacts":{"mesh_path":str(MESH.relative_to(ROOT)),"mesh_sha256":mesh_sha,"mesh_bytes":MESH.stat().st_size,
       "label_path":str(LABELS.relative_to(ROOT)),"label_sha256":label_sha,"original_tumor_path":str(TUMOR_SOURCE.relative_to(ROOT)),"original_tumor_sha256":tumor_sha,
       "phase3b_report_path":str(P3_REPORT.relative_to(ROOT)),"phase4b_generation_report_path":str(P4_REPORT.relative_to(ROOT))},
      "reproducibility":{"git_commit":os.popen("git -C '"+str(ROOT)+"' rev-parse HEAD 2>/dev/null").read().strip() or "NOT_AVAILABLE","python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"nibabel":nib.__version__,"mesh_reader":"standalone Python MSH 2.2 ASCII parser in this certification tool","tool_sha256":sha(Path(__file__))},
      "image":{"shape":list(shape),"spacing_mm":list(img.header.get_zooms()[:3]),"orientation":list(nib.aff2axcodes(affine)),"affine":affine.tolist(),"padded_affine":affine.tolist(),"affine_determinant":float(np.linalg.det(affine[:3,:3])),"handedness":"right-handed" if np.linalg.det(affine[:3,:3])>0 else "left-handed","voxel_volume_mm3":voxel_volume,"padding_each_side":pad,"label_counts":label_counts,"invalid_values":invalid_label_values,"physical_occupied_bbox_world":{"min":image_bbox.min(axis=0).tolist(),"max":image_bbox.max(axis=0).tolist()},"phase3b_original_shape":p3["original_shape"],"phase3b_original_affine":p3["original_affine"],"phase3b_final_affine":p3["final_affine"]},
      "mesh":{"format_version":version,"file_type":"ASCII" if version and version.startswith("2.") else "see MSH format header","data_size_bytes":8,"vertices":len(xyz),"hex8":len(hexes),"quad4":len(quads),"hex8_by_attribute":hcnt,"quad4_by_attribute":qcnt,"physical_names":{f"{d}:{t}":n for (d,t),n in physical.items()},"unexpected_element_types":unexpected,"unexpected_attributes":{"volume":sorted(set(hcnt)-set(range(1,6))),"boundary":sorted(set(qcnt)-set(range(101,106)))},"mesh_bbox_world":mesh_bbox,"referenced_vertex_count":len(referenced),"unused_vertices":len(xyz)-len(referenced),"duplicate_hexes":duplicate_hexes,"repeated_node_hexes":repeated_nodes,"duplicate_coordinate_vertices":len(xyz)-unique_coordinate_count},
      "geometry_quality":geom,"face_topology":{"unique_volume_faces":len(starts),"internal_faces":internal,"exterior_faces":exterior,"nonmanifold_faces":nonmanifold,"zero_owner_faces":zero_owner,"crack_check":"all exported coordinate triples unique; every shared canonical face uses exact shared node IDs; all cross-material/exterior faces matched against QUAD4 keys","material_adjacency_faces":adj_text,"same_material_internal_faces":same_internal,"forbidden_cross_material_faces":forbidden_count,"exterior_faces_by_material":exterior_by_mat,"quad_orphans":orphan,"duplicate_quads":qdup,"quad_semantic_errors":q_bad_sem,"missing_expected_quads":missing_quads,"unexpected_quads":unexpected_quads},
      "materials":material_summary,"tumor":tumor,"tumor_source_vs_label_grid":tumor_img_vs_labels,"original_tumor_image":{"shape":list(tumor_img.shape),"spacing_mm":list(tumor_img.header.get_zooms()[:3]),"orientation":list(nib.aff2axcodes(tumor_img.affine)),"affine":np.asarray(tumor_img.affine).tolist(),"affine_determinant":float(np.linalg.det(tumor_img.affine[:3,:3]))},
      "background_topology":bg_top,"image_components":img_comps,"mesh_components":mesh_components,
      "interfaces":{"area_and_normals":quad_area_summary,"surface_components":surface_components,"invalid_quads":invalid_quad},
      "image_mesh_reconstruction":{"voxels_compared":int(labels.size),"occupied_source_voxels":int(occ.sum()),"occupied_reconstructed_voxels":int(np.count_nonzero(reconstructed)),"different_voxels":int(np.count_nonzero(diff)),"confusion_matrix_rows_source_0_to_5_cols_mesh_0_to_5":np.bincount((labels.astype(np.int64)*6+reconstructed.astype(np.int64)).ravel(),minlength=36).reshape(6,6).tolist(),"per_material_dice_jaccard":{str(m):{"dice":float(2*np.count_nonzero((labels==m)&(reconstructed==m))/(np.count_nonzero(labels==m)+np.count_nonzero(reconstructed==m))),"jaccard":float(np.count_nonzero((labels==m)&(reconstructed==m))/np.count_nonzero((labels==m)|(reconstructed==m)))} for m in range(1,6)},"mesh_material_mapping_disagreements":coordinate_map_bad,"duplicate_voxel_mappings":int(len(hexes)-unique_cells),"max_inverse_affine_integer_residual":coord_error,"max_node_corner_coordinate_discrepancy_mm":max_coord_discrepancy,"landmarks":landmark_records},
      "volume_conservation":{"source_total_mm3":float(occ.sum()*voxel_volume),"mesh_total_mm3":float(vols.sum()),"absolute_difference_mm3":float(abs(occ.sum()*voxel_volume-vols.sum())),"relative_difference":float(abs(occ.sum()*voxel_volume-vols.sum())/max(occ.sum()*voxel_volume,1)),"method_a":"material label voxel counts x abs(det(NIfTI affine linear block))","method_b":"sum of signed affine HEX8 Jacobian determinants"},
      "gates":gates,"audit":{"wall_seconds":time.perf_counter()-T0,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"final_rss_bytes":rss_bytes()},
      "scope_statement":"This certification establishes geometric, topological, material-label, interface, coordinate, and image-to-mesh fidelity of the authoritative computational mesh. It does not make claims about solution accuracy, discretization convergence, constitutive model validity, or biomechanical prediction accuracy.",
      "methods_summary":"A patient-specific conforming five-region voxel-based hexahedral computational mesh was generated from the MRI-derived material segmentation. Independent mesh certification demonstrated valid and non-degenerate hexahedral geometry, conforming volumetric connectivity, exact material and interface reconciliation, valid physical-coordinate mapping, absence of forbidden material contacts and unintended internal cavities, and exact preservation of the original tumor segmentation. Reverse reconstruction of the material-label image from the exported mesh confirmed image-to-mesh material fidelity."}
    def json_default(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(f"not JSON serializable: {type(obj).__name__}")
    (OUT/"authoritative_mesh_certificate.json").write_text(json.dumps(report,indent=2,allow_nan=False,default=json_default)+"\n")
    failed=[k for k,v in gates.items() if not v]
    summary=["AUTHORITATIVE FULL-RESOLUTION MESH CERTIFICATION", "="*60,
      f"CLASSIFICATION: {report['classification']}",f"MESH_CERTIFIED = {'YES' if certified else 'NO'}",f"MESHING_MVP = {'COMPLETE' if certified else 'NOT_CLOSED'}",
      "",f"Mesh: {MESH.relative_to(ROOT)}",f"SHA256: {mesh_sha}",f"Bytes: {MESH.stat().st_size:,}",f"Source labels: {LABELS.relative_to(ROOT)}",f"Label SHA256: {label_sha}",f"Original tumor: {TUMOR_SOURCE.relative_to(ROOT)}",f"Tumor SHA256: {tumor_sha}",
      "",f"MSH {version}; vertices={len(xyz):,}; HEX8={len(hexes):,}; QUAD4={len(quads):,}",f"HEX by material: {hcnt}",f"QUAD by interface: {qcnt}",
      f"HEX volume min/max/mean: {geom['volume']['min']:.12g} / {geom['volume']['max']:.12g} / {geom['volume']['mean']:.12g} mm^3",f"Jacobian min/max: {geom['signed_jacobian_center']['min']:.12g} / {geom['signed_jacobian_center']['max']:.12g}",
      f"Scaled Jacobian min/max: {geom['scaled_jacobian_vertex']['min']:.12g} / {geom['scaled_jacobian_vertex']['max']:.12g}",f"Edges min/max: {geom['edge_length']['min']:.12g} / {geom['edge_length']['max']:.12g} mm",f"Max aspect ratio: {geom['maximum_aspect_ratio']:.12g}",
      f"Volume faces: unique={len(starts):,}, internal={internal:,}, exterior={exterior:,}, nonmanifold={nonmanifold}",f"Material adjacency: {adj_text}",f"Forbidden adjacencies: {forbidden_count}",
      f"QUAD checks: orphan={orphan}, duplicate={qdup}, semantic={q_bad_sem}, missing={missing_quads}, unexpected={unexpected_quads}, invalid geometry={invalid_quad}",
      f"Tumor exact={tumor['exact']}; FN={tumor['false_negative']}; FP={tumor['false_positive']}; Dice={tumor['dice']:.12g}; Jaccard={tumor['jaccard']:.12g}; centroid delta={tumor['centroid_difference_mm']:.12g} mm",
      f"Image reconstruction differing voxels: {int(np.count_nonzero(diff)):,}; max node discrepancy={max_coord_discrepancy:.12g} mm",f"Internal background: {bg_top['internal_components']} components, {bg_top['internal_voxels']} voxels",
      f"Audit elapsed: {report['audit']['wall_seconds']:.1f}s; peak RSS={report['audit']['peak_rss_bytes']/1e9:.3f} GB", "", "GATES:"]
    summary += [f"  {'PASS' if v else 'FAIL'} {k}" for k,v in gates.items()]
    summary += ["", "This certification establishes geometric, topological, material-label, interface, coordinate, and image-to-mesh fidelity of the authoritative computational mesh. It does not make claims about solution accuracy, discretization convergence, constitutive model validity, or biomechanical prediction accuracy.", "", "Methods/Validation summary: " + report["methods_summary"]]
    if failed: summary += ["", "FAILED INVARIANTS: " + ", ".join(failed)]
    (OUT/"authoritative_mesh_certificate.txt").write_text("\n".join(summary)+"\n")
    stage(f"certification finished: {report['classification']}")
    print("\n".join(summary),flush=True)
    return 0 if certified else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"CERTIFICATION ABORTED: {type(e).__name__}: {e}",file=sys.stderr,flush=True)
        raise
