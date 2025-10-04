#import nibabel as nib
#
#img_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"
#mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"
#
#
#img = nib.load(img_path)
#mask = nib.load(mask_path)
#
#print("Image shape:", img.shape)
#print("Mask shape:", mask.shape)
#
#print("Image affine:\n", img.affine)
#print("Mask affine:\n", mask.affine)


import nibabel as nib
from skimage import measure
import numpy as np
import pyvista as pv  # Optional for exporting mesh

# Load mask
mask_img = nib.load('/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz')
mask_data = mask_img.get_fdata()
affine = mask_img.affine

# Get vertices and faces
verts, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)

# Transform verts to world coordinates
verts_world = nib.affines.apply_affine(affine, verts)

# Export to VTK (optional)
faces = np.hstack([[3, *face] for face in faces])  # Convert to flat format for VTK
mesh = pv.PolyData(verts_world, faces)
mesh.save("tumor.vtk")
