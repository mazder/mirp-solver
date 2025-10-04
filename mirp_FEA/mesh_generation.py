import os
import numpy as np
import SimpleITK as sitk
import pyvista as pv
import trimesh
import gmsh
import os
from glob import glob
import meshio
import shutil
import pyvista as pv
from contextlib import redirect_stdout, redirect_stderr

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

def get_image_mask_pairs(root_dir):
    """
    Given a root directory, returns a list of (image_path, mask_path) tuples
    for the modalities 1_T1, 2_T1GD, 3_T2, 4_FLAIR, each paired with 5_MASK.
    """
    image_subfolders = ['1_T1', '2_T1GD', '3_T2', '4_FLAIR']
    pairs = []

    patient_dirs = sorted(glob(os.path.join(root_dir, 'EGD-*')))

    for patient_dir in patient_dirs:
        mr_dirs = glob(os.path.join(patient_dir, 'MR_EGD-*'))
        if not mr_dirs:
            continue
        mr_dir = mr_dirs[0]

        mask_files = glob(os.path.join(mr_dir, '5_MASK/NIFTI/*.nii.gz'))
        if not mask_files:
            continue
        mask_path = mask_files[0]

        for subfolder in image_subfolders:
            image_files = glob(os.path.join(mr_dir, f'{subfolder}/NIFTI/*.nii.gz'))
            if not image_files:
                continue
            image_path = image_files[0]
            pairs.append((image_path, mask_path, f'{mr_dir}/{subfolder}'))

    return pairs

# ----------------------------
# Debugging Function
# ----------------------------
def debugging():
    #img_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"
    #mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"

    img_path = "/workspace/mirp-solver/simulation_data/sub-verse030/im/sub-verse030_vertebra_23_im.nii.gz"
    mask_path = "/workspace/mirp-solver/simulation_data/sub-verse030/seg/sub-verse030_vertebra_23_seg.nii.gz"

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

    mesh = meshio.read('tumor_mesh_3d.msh')
    meshio.write('tumor_mesh_3d.vtu', mesh)


# Helper to log clean messages
def make_logger(log_file_path, also_print=True):
    def log_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
        if also_print:
            print(message, **kwargs)
    return log_print


def slice_mesh_to_png(mesh_file, output_png, origin=None, normal='z', show_edges=True):
    """
    Load a .vtu or .msh mesh, slice it along a given plane, and save the slice as a PNG image.

    Parameters:
        mesh_file (str): Path to .msh or .vtu file.
        output_png (str): Output PNG file path.
        origin (tuple or None): Origin point of the slicing plane. If None, center is used.
        normal (str or tuple): Normal direction for slicing ('x', 'y', 'z' or a 3D vector).
        show_edges (bool): Whether to show edges in the plot.
    """
    # Convert .msh to .vtu if necessary
    if mesh_file.endswith('.msh'):
        mesh = meshio.read(mesh_file)
        temp_vtu = mesh_file.replace('.msh', '.vtu')
        meshio.write(temp_vtu, mesh)
        mesh_file = temp_vtu

    # Load mesh into PyVista
    pv_mesh = pv.read(mesh_file)

    # Define slice plane
    if origin is None:
        origin = pv_mesh.center

    slice_plane = pv_mesh.slice(normal=normal, origin=origin)

    # Plot to file (off_screen=True avoids opening GUI)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(slice_plane, show_edges=show_edges)
    plotter.show(screenshot=output_png)
    print(f"✅ Slice saved to: {output_png}")

# ----------------------------
# MAIN PIPELINE
# ----------------------------

if __name__ == "__main__":

    debugging()

#    root = '/workspace/mirp-solver/simulation_data/brain_tumar'
#    pairs = get_image_mask_pairs(root)
#
#    for img_path, mask_path, mesh_dir in pairs:
#        mesh_path = os.path.join(mesh_dir, 'mesh')
#        if os.path.exists(mesh_path):
#            shutil.rmtree(mesh_path)
#        os.makedirs(mesh_path)
#
#        # Log files
#        stat_log_path = os.path.join(mesh_path, 'stat.log')           # Clean log
#        debug_log_path = os.path.join(mesh_path, 'full_debug.log')    # Full output log
#
#        log_print = make_logger(stat_log_path)
#
#        # Redirect all stdout/stderr to full_debug.log
#        with open(debug_log_path, 'w') as debug_log:
#            with redirect_stdout(debug_log), redirect_stderr(debug_log):
#
#                # Now only log_print() will write to stat.log
#                log_print("Image:", img_path)
#                log_print("Mask :", mask_path)
#                log_print("Mesh :", mesh_path)
#                log_print("---")
#
#                image, mask = load_nifti(img_path, mask_path)
#                log_print("Original image size:", image.GetSize())
#
#                image, mask = resample_isotropic(image, mask)
#                log_print("Resampled size:", image.GetSize())
#
#                image = normalize_intensity(image)
#                image, mask = crop_to_mask(image, mask)
#                log_print("Cropped size:", image.GetSize())
#
#                sitk.WriteImage(image, os.path.join(mesh_path, 'preprocessed_image.nii.gz'))
#                sitk.WriteImage(mask, os.path.join(mesh_path, 'preprocessed_mask.nii.gz'))
#                log_print("✅ Preprocessing complete, files saved.")
#
#                stl_file = mask_to_stl(mask, stl_file=os.path.join(mesh_path, 'tumor_surface.stl'))
#                repaired_stl = repair_stl(stl_file, output_stl=os.path.join(mesh_path, 'tumor_surface_final.stl'))
#
#                generate_gmsh_mesh_geo(repaired_stl, msh_file=os.path.join(mesh_path, 'tumor_mesh_3d.msh'), fine=True)
#                log_print("✅ Mesh generation complete.")
#
#                log_print("========== Mesh Conversion ===========================")
#                mesh = meshio.read(os.path.join(mesh_path, 'tumor_mesh_3d.msh'))
#                meshio.write(os.path.join(mesh_path, 'tumor_mesh_3d.vtu'), mesh)
#                log_print("✅ Mesh conversion to .vtu complete.")
#                log_print("======================================================")
#
#                mesh_stl = pv.read(os.path.join(mesh_path, 'tumor_surface_final.stl'))
#                bounds = mesh_stl.bounds
#                log_print(f"Bounds in mm: {bounds}")
#
#                size_m = [
#                    bounds[1] - bounds[0],
#                    bounds[3] - bounds[2],
#                    bounds[5] - bounds[4]
#                ]
#                log_print(f"Mesh size (in mm): X={size_m[0]:.4f}, Y={size_m[1]:.4f}, Z={size_m[2]:.4f}")
#
#                slice_mesh_to_png(os.path.join(mesh_path, 'tumor_mesh_3d.vtu'), os.path.join(mesh_path, 'slice_output.png'), normal='z')
#
#                #mesh_to_view = pv.read(os.path.join(mesh_path, 'tumor_mesh_3d.msh'))
#                #mesh_to_view.plot(show_edges=True)

