import os
import numpy as np
import nibabel as nib
from skimage import measure
import pyvista as pv
import meshio
import subprocess

def extract_surface_from_nii(nii_path, label=0, level=0.5, name="output"):
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine

    if label > 0:
        data = (data == label).astype(np.uint8)

    verts_vox, faces, _, _ = measure.marching_cubes(data, level=level)
    verts_world = nib.affines.apply_affine(affine, verts_vox)
    faces_pv = np.hstack([[3, *face] for face in faces])

    mesh = pv.PolyData(verts_world, faces_pv)
    out_file = f"{name}.vtk"
    mesh.save(out_file)
    print(f"[✓] Saved surface mesh: {out_file} with bounds {mesh.bounds}")
    return mesh

def write_surface_stl_from_mask(mask_path, stl_file="surface.stl"):
    nii = nib.load(mask_path)
    data = nii.get_fdata()
    affine = nii.affine

    verts_vox, faces, _, _ = measure.marching_cubes(data > 0.5, level=0.5)
    verts_world = nib.affines.apply_affine(affine, verts_vox)

    meshio_mesh = meshio.Mesh(points=verts_world, cells=[("triangle", faces)])
    meshio.write(stl_file, meshio_mesh)
    print(f"[✓] Saved surface STL: {stl_file}")
    return stl_file

def write_gmsh_geo(surface_file="surface.stl", geo_file="tet.geo", lc=1.0):
    geo_dir = os.path.dirname(os.path.abspath(geo_file))
    surface_rel = os.path.relpath(os.path.abspath(surface_file), start=geo_dir).replace("\\", "/")

    geo_text = f"""
Merge "{surface_rel}";

Surface Loop(1) = {{1}};
Volume(1) = {{1}};

Physical Volume("domain") = {{1}};
Physical Surface("boundary") = {{1}};

Mesh.Algorithm3D = 4;
Mesh.CharacteristicLengthMax = {lc};
"""
    with open(geo_file, "w") as f:
        f.write(geo_text.strip())
    print(f"[✓] Wrote Gmsh geo file: {geo_file}")
    return geo_file

def generate_tet_mesh(geo_file="tet.geo", out_file="tet_mesh.msh"):
    cmd = ["gmsh", "-3", geo_file, "-format", "msh2", "-o", out_file]
    print("🚀 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[✓] Generated tetrahedral mesh: {out_file}")
    return out_file

def convert_msh_to_vtu(msh_file="tet_mesh.msh", vtu_file="tet_mesh.vtu"):
    mesh = meshio.read(msh_file)
    meshio.write(vtu_file, mesh)
    print(f"[✓] Converted mesh to VTU: {vtu_file}")
    return vtu_file

def read_and_transform_tet_mesh(vtu_file, affine, apply_affine=False):
    mesh = meshio.read(vtu_file)
    if apply_affine:
        mesh.points = nib.affines.apply_affine(affine, mesh.points)
        print("[✓] Applied affine transformation to tetrahedral mesh.")
    meshio.write(vtu_file, mesh)
    tet_mesh_pv = pv.read(vtu_file)
    print(f"Tetrahedral mesh bounds: {tet_mesh_pv.bounds}")
    return tet_mesh_pv

def visualize_all(brain_mesh, tumor_mesh, tet_mesh):
    p = pv.Plotter()
    p.add_mesh(brain_mesh, color="lightgray", opacity=0.3, label="Brain Surface")
    p.add_mesh(tumor_mesh, color="orange", opacity=0.5, label="Tumor Surface")
    p.add_mesh(tet_mesh, color="red", opacity=0.3, label="Tetrahedral Mesh")
    p.add_legend()
    p.show()

if __name__ == "__main__":
    brain_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"
    tumor_mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Create tumor surface STL for mesh input
    stl_file = os.path.join(output_dir, "tumor_surface.stl")
    stl_file = write_surface_stl_from_mask(tumor_mask_path, stl_file)

    # Step 2: Compute refined characteristic length
    nii = nib.load(tumor_mask_path)
    spacing = nii.header.get_zooms()
    lc = min(spacing) * 0.25  # 4x finer mesh than default
    print(f"🔍 Using refined characteristic length for 18M mesh: {lc}")

    # Step 3: Write geo file
    geo_file = os.path.join(output_dir, "tet.geo")
    geo_file = write_gmsh_geo(stl_file, geo_file, lc)

    # Step 4: Run Gmsh tetra meshing
    msh_file = os.path.join(output_dir, "tet_mesh.msh")
    msh_file = generate_tet_mesh(geo_file, msh_file)

    # Step 5: Convert msh to vtu for PyVista
    vtu_file = os.path.join(output_dir, "tet_mesh.vtu")
    vtu_file = convert_msh_to_vtu(msh_file, vtu_file)

    # Step 6: Extract brain and tumor surfaces
    brain_mesh = extract_surface_from_nii(brain_path, label=0, level=0.5, name=os.path.join(output_dir, "brain_surface"))
    tumor_mesh = extract_surface_from_nii(tumor_mask_path, label=1, level=0.5, name=os.path.join(output_dir, "tumor_surface"))

    # Step 7: Read tetra mesh without re-applying affine (already done)
    tet_mesh = read_and_transform_tet_mesh(vtu_file, nii.affine, apply_affine=False)

    # Step 8: Visualize all together
    visualize_all(brain_mesh, tumor_mesh, tet_mesh)
