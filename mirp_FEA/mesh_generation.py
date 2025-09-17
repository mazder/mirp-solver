import os
import numpy as np
import SimpleITK as sitk
import pyvista as pv
import trimesh
import gmsh

# ----------------------------
# Preprocessing Functions
# ----------------------------
def load_nifti(image_path, mask_path):
    """Load NIfTI image and mask with SimpleITK"""
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    return image, mask

def resample_isotropic(image, mask, new_spacing=(1.0, 1.0, 1.0)):
    """Resample image and mask to isotropic voxel spacing"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampled_image = resampler.Execute(image)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For masks
    resampled_mask = resampler.Execute(mask)

    return resampled_image, resampled_mask

def normalize_intensity(image, min_val=0.0, max_val=1.0):
    """Normalize intensity to [min_val, max_val]"""
    img_array = sitk.GetArrayFromImage(image).astype(np.float32)
    img_norm = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)
    img_norm = img_norm * (max_val - min_val) + min_val
    return sitk.GetImageFromArray(img_norm)

def crop_to_mask(image, mask, margin=5):
    """Crop image and mask around non-zero mask region (safe clamped version)."""
    mask_array = sitk.GetArrayFromImage(mask)
    coords = np.argwhere(mask_array)

    if coords.size == 0:
        print("⚠️ Warning: Mask is empty, skipping crop.")
        return image, mask

    min_coords = coords.min(axis=0) - margin
    max_coords = coords.max(axis=0) + margin

    # Clamp to valid range
    size_z, size_y, size_x = mask_array.shape
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, [size_z - 1, size_y - 1, size_x - 1])

    # Convert to size and index for SimpleITK
    start_index = [int(min_coords[2]), int(min_coords[1]), int(min_coords[0])]  # z,y,x → x,y,z
    size = [int(max_coords[2] - min_coords[2] + 1),
            int(max_coords[1] - min_coords[1] + 1),
            int(max_coords[0] - min_coords[0] + 1)]

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_index)
    roi_filter.SetSize(size)

    cropped_image = roi_filter.Execute(image)
    cropped_mask = roi_filter.Execute(mask)

    return cropped_image, cropped_mask

# ----------------------------
# Mesh Functions
# ----------------------------
def mask_to_stl(mask_image, stl_file="tumor_surface.stl"):
    mask_array = sitk.GetArrayFromImage(mask_image)
    grid = pv.wrap(mask_array.astype(np.uint8).transpose(2,1,0))
    mesh = grid.contour(isosurfaces=[0.5])
    mesh.save(stl_file)
    return stl_file

#def mask_to_stl(mask_image, stl_file="tumor_surface.stl"):
#    import pyvista as pv
#    import numpy as np
#    import SimpleITK as sitk
#
#    # Get mask array and spacing (in mm)
#    mask_array = sitk.GetArrayFromImage(mask_image)  # Shape: (z, y, x)
#    spacing_mm = mask_image.GetSpacing()             # Spacing in mm
#    spacing_m = tuple(s / 1000.0 for s in spacing_mm)  # Convert to meters
#
#    # Create pyvista grid with correct size
#    grid = pv.ImageData()
#    grid.dimensions = np.array(mask_array.shape[::-1])  # Shape must be (x, y, z)
#    grid.origin = (0.0, 0.0, 0.0)
#    grid.spacing = spacing_m
#
#    # Assign the mask values as **point data** (required by contour)
#    grid.point_data["values"] = mask_array.transpose(2, 1, 0).flatten(order="F")
#
#    # Extract surface via contour at threshold 0.5
#    mesh = grid.contour(isosurfaces=[0.5])
#    mesh.save(stl_file)
#    return stl_file


def repair_stl(input_stl, output_stl="tumor_surface_final.stl"):
    tmesh = trimesh.load_mesh(input_stl)
    # Keep largest connected component
    components = tmesh.split(only_watertight=False)
    tmesh = max(components, key=lambda x: x.volume if x.is_watertight else x.area)
    tmesh.remove_duplicate_faces()
    tmesh.remove_degenerate_faces()
    tmesh.fill_holes()
    tmesh.remove_unreferenced_vertices()
    tmesh.process(validate=True)
    tmesh.export(output_stl)
    return output_stl

##def generate_gmsh_mesh_geo(stl_file, msh_file="tumor_mesh_3d.msh"):
#def generate_gmsh_mesh_geo(stl_file, msh_file="tumer_mesh_3d.msh"):
#    gmsh.initialize()
#    gmsh.option.setNumber("General.Terminal", 1)
#    gmsh.model.add("tumor_mesh")
#
#    gmsh.merge(stl_file)
#
#    surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
#
#    # Create physical group for boundaries (all surfaces)
#    gmsh.model.addPhysicalGroup(2, surfaces, 1)
#    gmsh.model.setPhysicalName(2, 1, "tumor_boundary")
#
#    loop_tag = gmsh.model.geo.addSurfaceLoop(surfaces)
#    vol_tag = gmsh.model.geo.addVolume([loop_tag])
#    gmsh.model.geo.synchronize()
#
#    gmsh.model.addPhysicalGroup(3, [vol_tag], 2)
#    gmsh.model.setPhysicalName(3, 2, "tumor_volume")
#
#    gmsh.model.mesh.generate(3)
#
#    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
#    gmsh.option.setNumber("Mesh.Binary", 0)
#
#    gmsh.write(msh_file)
#    gmsh.finalize()
#    print(f"✅ 3D GEO mesh saved in MSH v2.2: {msh_file}")


#works
# ----------------------------
# Step 3: Generate 3D mesh (GEO kernel)
# ----------------------------
#def generate_gmsh_mesh_geo(stl_file, msh_file="tumor_mesh_3d.msh"):
#    gmsh.initialize()
#    gmsh.option.setNumber("General.Terminal", 1)
#    gmsh.model.add("tumor_mesh")
#
#    gmsh.merge(stl_file)
#
#    # Build discrete surfaces
#    surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
#    loop_tag = gmsh.model.geo.addSurfaceLoop(surfaces)
#    vol_tag = gmsh.model.geo.addVolume([loop_tag])
#
#    # ✅ Add Physical Volume so MFEM gets attribute > 0
#    gmsh.model.addPhysicalGroup(3, [vol_tag], 1)
#    gmsh.model.setPhysicalName(3, 1, "tumor")
#
#    gmsh.model.geo.synchronize()
#
#    # Mesh
#    gmsh.model.mesh.generate(3)
#
#    # Force MFEM-compatible format
#    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
#    gmsh.option.setNumber("Mesh.Binary", 0)
#
#    gmsh.write(msh_file)
#    gmsh.finalize()
#    print(f"✅ 3D GEO mesh saved in MSH v2.2: {msh_file}")

def generate_gmsh_mesh_geo(stl_file, msh_file="tumor_mesh_3d.msh", fine=True):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("tumor_mesh")
    gmsh.merge(stl_file)

    surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.model.setPhysicalName(2, 1, "tumor_boundary")

    loop_tag = gmsh.model.geo.addSurfaceLoop(surfaces)
    vol_tag = gmsh.model.geo.addVolume([loop_tag])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [vol_tag], 2)
    gmsh.model.setPhysicalName(3, 2, "tumor_volume")

    if fine:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.5)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(msh_file)
    gmsh.finalize()
    print(f"✅ 3D mesh saved in MSH v2.2: {msh_file}")


# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":

    img_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"
    mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"

    #img_path = "/workspace/mirp-solver/platform/test_cases/nii_mesh_mfem/sub-verse030/im/sub-verse030_vertebra_23_im.nii.gz"
    #mask_path = "/workspace/mirp-solver/platform/test_cases/nii_mesh_mfem/sub-verse030/seg/sub-verse030_vertebra_23_seg.nii.gz"

    # 1. Load
    image, mask = load_nifti(img_path, mask_path)
    print("Original image size:", image.GetSize())

    # 2. Resample to isotropic spacing
    image, mask = resample_isotropic(image, mask)
    print("Resampled size:", image.GetSize())

    # 3. Normalize intensity
    image = normalize_intensity(image)

    # 4. Crop to mask region
    image, mask = crop_to_mask(image, mask)
    print("Cropped size:", image.GetSize())

    # Save preprocessed outputs
    sitk.WriteImage(image, "preprocessed_image.nii.gz")
    sitk.WriteImage(mask, "preprocessed_mask.nii.gz")
    print("✅ Preprocessing complete, files saved.")

    # 5. Generate mesh
    stl_file = mask_to_stl(mask)
    repaired_stl = repair_stl(stl_file)
    #generate_gmsh_mesh_geo(repaired_stl)
    generate_gmsh_mesh_geo(repaired_stl, msh_file="tumor_mesh_3d.msh", fine=True)  # or fine=False
