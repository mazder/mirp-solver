#!/usr/bin/env python3
"""
inspect_mesh_mask.py

Compare a tetrahedral mesh (.msh or .vtu) with its original 3D mask (.nii.gz)
to check scaling, voxel consistency, and mesh validity.
Usage:
    python inspect_mesh_mask.py <mesh_file.msh|.vtu> <mask_file.nii.gz>
"""

import sys
import numpy as np
import meshio
import SimpleITK as sitk

def inspect_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask)
    spacing = np.array(mask.GetSpacing())[::-1]   # (z,y,x)
    origin = np.array(mask.GetOrigin())[::-1]
    mask_size = np.array(mask.GetSize())          # voxels
    mask_physical = mask_size * spacing

    print(f"🔍 Inspecting mask: {mask_path}")
    print("✅ Mask loaded.")
    print(f"   - Shape (z,y,x): {arr.shape}")
    print(f"   - Spacing: {spacing}")
    print(f"   - Origin: {origin}")
    print(f"   - Physical size: {mask_physical}")
    print(f"   - Unique labels: {np.unique(arr)}\n")
    return mask_physical, origin

def inspect_mesh(mesh_file):
    print(f"🔍 Inspecting mesh: {mesh_file}")
    mesh = meshio.read(mesh_file)
    points = mesh.points
    cells = mesh.cells_dict
    print("✅ Successfully read mesh.")
    print(f"   - Points: {len(points)}")
    print(f"   - Cells: {list(cells.keys())}")
    if "tetra" in cells:
        print(f"   - Tetrahedra: {len(cells['tetra'])}")
    else:
        print("❌ No tetrahedral cells found!")

    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    mesh_physical_size = max_bounds - min_bounds
    print(f"   - Bounding box: {min_bounds} → {max_bounds}")
    print(f"   - Physical size: {mesh_physical_size}\n")
    return mesh_physical_size, min_bounds

def compare_mesh_mask(mesh_file, mask_file):
    mask_physical, mask_origin = inspect_mask(mask_file)
    mesh_physical_size, mesh_min = inspect_mesh(mesh_file)

    size_diff = mask_physical - mesh_physical_size
    origin_diff = mask_origin - mesh_min

    print("📐 Consistency check with mask:")
    print(f"   - Mask size: {mask_physical}")
    print(f"   - Mesh size: {mesh_physical_size}")
    print(f"   - Size difference: {size_diff}")
    print(f"   - Origin difference: {origin_diff}")

    if np.allclose(mesh_physical_size, mask_physical, rtol=0.05) and np.allclose(mesh_min, mask_origin, rtol=0.05):
        print("🎉 Mesh matches mask physical size and origin within 5% tolerance.")
    else:
        print("⚠️ Mesh does not perfectly match mask! Check scaling or origin.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_mesh_mask.py <mesh_file.msh|.vtu> <mask_file.nii.gz>")
        sys.exit(1)

    mesh_file = sys.argv[1]
    mask_file = sys.argv[2]
    compare_mesh_mask(mesh_file, mask_file)

#
#  Info    : 0 open elements
#Info    : Success !
#Info    : 2723299 points, 18338156 elements
#Info    : Done meshing 3D (Wall 1488.38s, CPU 1429.88s)
#Info    : Optimizing mesh...
#Info    : Optimizing volume 1
#Info    : Optimization starts (volume = 2.57512e+06) with worst = 7.12491e-13 / average = 0.624711:
#Info    : 0.00 < quality < 0.10 :      1938 elements
#Info    : 0.10 < quality < 0.20 :     29936 elements
#Info    : 0.20 < quality < 0.30 :    558814 elements
#Info    : 0.30 < quality < 0.40 :   1779882 elements
#Info    : 0.40 < quality < 0.50 :   2605005 elements
#Info    : 0.50 < quality < 0.60 :   3116972 elements
#Info    : 0.60 < quality < 0.70 :   3390156 elements
#Info    : 0.70 < quality < 0.80 :   3265119 elements
#Info    : 0.80 < quality < 0.90 :   2604766 elements
#Info    : 0.90 < quality < 1.00 :    985568 elements
#Info    : 436900 edge swaps, 51056 node relocations (volume = 2.57512e+06): worst = 2.12764e-11 / average = 0.636207 (Wall 16.6162s, CPU 16.6066s)
#Info    : 473036 edge swaps, 62109 node relocations (volume = 2.57512e+06): worst = 2.12764e-11 / average = 0.637 (Wall 22.6511s, CPU 22.6421s)
#Warning : 12 ill-shaped tets are still in the mesh
#Info    : 0.00 < quality < 0.10 :       174 elements
#Info    : 0.10 < quality < 0.20 :       974 elements
#Info    : 0.20 < quality < 0.30 :     51014 elements
#Info    : 0.30 < quality < 0.40 :   1742646 elements
#Info    : 0.40 < quality < 0.50 :   2601186 elements
#Info    : 0.50 < quality < 0.60 :   3142229 elements
#Info    : 0.60 < quality < 0.70 :   3496621 elements
#Info    : 0.70 < quality < 0.80 :   3357499 elements
#Info    : 0.80 < quality < 0.90 :   2620142 elements
#Info    : 0.90 < quality < 1.00 :    982718 elements
#Info    : Done optimizing mesh (Wall 58.0395s, CPU 56.6866s)
#Info    : 2723279 nodes 18172715 elements
#Warning : ------------------------------
#Warning : Mesh generation error summary
#Warning :     1 warning
#Warning :     0 errors
#Warning : Check the full log for details
#Warning : ------------------------------
#Info    : Writing 'tet_mesh.msh'...
#Info    : Done writing 'tet_mesh.msh'
#Info    : Stopped on Wed Oct  1 21:42:45 2025 (From start: Wall 1583.84s, CPU 1523.9s)
#✅ Generated tetrahedral mesh: tet_mesh.msh
#
#✅ Wrote tet_mesh.vtu for ParaView
#root@6e1f302d6e16:/workspace/mirp-solver/mirp_FEA# python3 inspect_mesh.py tet_mesh.msh /workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz
#🔍 Inspecting mask: /workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz
#✅ Mask loaded.
#   - Shape (z,y,x): (189, 233, 197)
#   - Spacing: [1. 1. 1.]
#   - Origin: [-72. 134.  98.]
#   - Physical size: [197. 233. 189.]
#   - Unique labels: [0 1]
#
#🔍 Inspecting mesh: tet_mesh.msh
#
#✅ Successfully read mesh.
#   - Points: 2723279
#   - Cells: ['triangle', 'tetra']
#   - Tetrahedra: 18050607
#   - Bounding box: [-72. 134.  98.] → [125. 367. 287.]
#   - Physical size: [197. 233. 189.]
#
#📐 Consistency check with mask:
#   - Mask size: [197. 233. 189.]
#   - Mesh size: [197. 233. 189.]
#   - Size difference: [0. 0. 0.]
#   - Origin difference: [0. 0. 0.]
#🎉 Mesh matches mask physical size and origin within 5% tolerance.
#

#Tetrahedral meshes were generated from segmented brain tumor masks in NIfTI format using a voxel-consistent approach. The binary masks were converted to closed surfaces via a marching cubes algorithm, preserving the physical dimensions of the original voxels. The resulting surfaces were imported into Gmsh, where the characteristic tetrahedral edge length was automatically set to match the smallest voxel dimension, ensuring the mesh resolution was consistent with the imaging data. Volumes were meshed using a Delaunay-based tetrahedralization, and surface and volume elements were assigned physical labels for boundary and domain identification. The generated meshes were exported in both .msh and .vtu formats, enabling simulation and visualization in ParaView. Mesh quality was verified to ensure geometric fidelity and proper alignment with the original mask.

#We generated patient-specific tetrahedral meshes from 3D MRI tumor segmentations using marching cubes to extract the tumor surface, followed by Gmsh-based tetrahedralization with voxel-consistent scaling. Linear elasticity simulations were performed in MFEM using first-order vector-valued H1 finite elements, with tumor or bone material properties assigned via Young’s modulus and Poisson’s ratio. Essential boundary conditions were applied on selected mesh boundaries, and body forces were included to model physiological loads. Element-wise strain and stress tensors were computed, and von Mises stress was projected for visualization. Surface curvature and normal traction were evaluated to estimate contraction ratios, while displacement and stress statistics, as well as average strain energy density, were calculated for global and tumor-specific analyses. All results were output in ParaView-compatible format for further post-processing and visualization.

