#!/usr/bin/env python3
"""
mesh_generation_tetra.py

Generate a tetrahedral mesh from a 3D medical image mask (.nii.gz),
with voxel-consistent scaling and automatic tetrahedron size selection.

Steps:
1. Convert the binary mask to a closed surface (STL) using marching cubes.
2. Automatically scale the surface to match mask physical dimensions.
3. Write a Gmsh .geo script to fill the volume with tetrahedra.
4. Automatically set the characteristic length based on voxel spacing.
5. Generate tetrahedral mesh with Gmsh.
6. Convert mesh to VTU format for visualization.

Requirements:
- SimpleITK
- scikit-image
- meshio
- Gmsh installed and in PATH
"""

import numpy as np
import SimpleITK as sitk
import meshio
import subprocess
from skimage import measure

def mask_to_surface(mask_path, surface_file="surface.stl"):
    """
    Convert a 3D mask to a closed STL surface using marching cubes.
    Automatically scales the surface to match mask physical size.
    Returns the surface file and mask spacing.
    """
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask)  # shape (Z,Y,X)
    spacing = np.array(mask.GetSpacing())[::-1]  # (z,y,x)
    origin = np.array(mask.GetOrigin())[::-1]

    verts, faces, normals, values = measure.marching_cubes(arr, level=0.5)

    # Normalize vertices to 0..1
    verts_norm = (verts - verts.min(axis=0)) / (verts.max(axis=0) - verts.min(axis=0))

    # Scale to mask physical size
    mask_size = np.array(mask.GetSize())        # in voxels
    mask_physical = mask_size * spacing
    points = verts_norm * mask_physical + origin

    mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])
    meshio.write(surface_file, mesh)
    print(f"✅ Saved scaled surface STL: {surface_file}")
    return surface_file, spacing


def write_geo(surface_file="surface_scaled.stl", geo_file="tet.geo", lc=1.0):
    """
    Write a Gmsh .geo file that wraps the surface and fills it with tetrahedra.
    """
    geo_text = f"""
    Merge "{surface_file}";

    Surface Loop(1) = {{1}};
    Volume(1) = {{1}};

    Physical Volume("domain") = {{1}};
    Physical Surface("boundary") = {{1}};

    Mesh.Algorithm3D = 4; // Delaunay
    Mesh.CharacteristicLengthMax = {lc};
    """
    with open(geo_file, "w") as f:
        f.write(geo_text)
    print(f"✅ Wrote Gmsh geo script: {geo_file}")
    return geo_file

def generate_tet_mesh(geo_file="tet.geo", out_file="tet_mesh.msh"):
    """
    Call Gmsh to generate tetrahedral mesh.
    """
    cmd = ["gmsh", "-3", geo_file, "-format", "msh2", "-o", out_file]
    print("🚀 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Generated tetrahedral mesh: {out_file}")
    return out_file

if __name__ == "__main__":
    mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"

    # Step 1: Extract surface and get mask spacing
    surface_file = "surface.stl"
    surface_file, spacing = mask_to_surface(mask_path, surface_file)

    # Step 2: Automatically set characteristic length (lc) from voxel spacing
    lc = spacing.min()  # smallest voxel dimension to capture all features
    print(f"📐 Using automatic characteristic length (lc) = {lc:.3f} mm based on mask voxel spacing {spacing}")

    # Step 3: Write Gmsh geo
    geo_file = write_geo(surface_file, "tet.geo", lc=lc)

    # Step 4: Generate tetrahedral mesh
    tet_file = generate_tet_mesh(geo_file, "tet_mesh.msh")

    # Step 5: Convert to VTU for ParaView
    msh = meshio.read(tet_file)
    meshio.write("tet_mesh.vtu", msh)
    print("✅ Wrote tet_mesh.vtu for ParaView")
