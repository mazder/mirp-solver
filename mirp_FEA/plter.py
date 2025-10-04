#import pyvista as pv
#import meshio
#
## --- Load meshes ---
#tet_mesh_file = "tet_mesh.vtu"
#hex_mesh_file = "hex_mesh.vtu"
#
## PyVista can read VTU files directly
#tet = pv.read(tet_mesh_file)
#hex_mesh = pv.read(hex_mesh_file)
#
## --- Create plotter ---
#plotter = pv.Plotter()
#plotter.add_mesh(tet, color='red', opacity=0.6, show_edges=True, label='Tetra Mesh')
#plotter.add_mesh(hex_mesh, color='blue', opacity=0.3, show_edges=True, label='Hex Mesh')
#
## --- Add legend and axes ---
#plotter.add_legend()
#plotter.show_axes()
#plotter.show_grid()
#plotter.background_color = 'white'
#
## --- Show interactive window ---
#plotter.show(title="Tet vs Hex Mesh Alignment")


#import meshio
#import numpy as np
#
#tet = meshio.read("tet_mesh.vtu")
#hex_mesh = meshio.read("hex_mesh.vtu")
#
#tet_points = tet.points
#hex_points = hex_mesh.points
#
#print("Tet mesh bounds:", tet_points.min(axis=0), tet_points.max(axis=0))
#print("Hex mesh bounds:", hex_points.min(axis=0), hex_points.max(axis=0))
#
## Compare if the differences are negligible
#diff_min = np.abs(tet_points.min(axis=0) - hex_points.min(axis=0))
#diff_max = np.abs(tet_points.max(axis=0) - hex_points.max(axis=0))
#print("Min differences:", diff_min)
#print("Max differences:", diff_max)


#import numpy as np
#import meshio
#
#tet = meshio.read("tet_mesh.vtu")
#hex_mesh = meshio.read("hex_mesh.vtu")
#
## Shift tet mesh to hex mesh origin
#shift = hex_mesh.points.min(axis=0) - tet.points.min(axis=0)
#tet_shifted = tet.points + shift
#
#print("New min corner diff:", tet_shifted.min(axis=0) - hex_mesh.points.min(axis=0))
#
## Compute approximate spacing along each axis
#tet_spacing = np.diff(np.unique(tet_shifted[:,0]))[0], np.diff(np.unique(tet_shifted[:,1]))[0], np.diff(np.unique(tet_shifted[:,2]))[0]
#hex_spacing = np.diff(np.unique(hex_mesh.points[:,0]))[0], np.diff(np.unique(hex_mesh.points[:,1]))[0], np.diff(np.unique(hex_mesh.points[:,2]))[0]
#
#print("Tet mesh spacing:", tet_spacing)
#print("Hex mesh spacing:", hex_spacing)
#print("Spacing difference:", np.array(tet_spacing) - np.array(hex_spacing))


#import pyvista as pv
#import meshio
#import numpy as np
#
## --- Load mesh function ---
#def load_mesh(filename):
#    mesh = meshio.read(filename)
#    if "tetra" in mesh.cells_dict:
#        cells = mesh.cells_dict["tetra"]
#        return pv.UnstructuredGrid({10: cells}, mesh.points)  # VTK_TETRA = 10
#    elif "hexahedron" in mesh.cells_dict:
#        cells = mesh.cells_dict["hexahedron"]
#        return pv.UnstructuredGrid({12: cells}, mesh.points)  # VTK_HEXAHEDRON = 12
#    else:
#        raise ValueError("No supported cells found in mesh")
#
## --- File paths ---
#tet_file = "tet_mesh.vtu"  # Tet mesh
#hex_file = "hex_mesh.vtu"  # Hex mesh
#
## --- Load meshes ---
#tet_grid = load_mesh(tet_file)
#hex_grid = load_mesh(hex_file)
#
## --- Align Tet mesh to Hex mesh origin ---
#shift = hex_grid.points.min(axis=0) - tet_grid.points.min(axis=0)
#tet_grid.points += shift
#print("Tet mesh shifted by:", shift)
#
## --- Print bounds for verification ---
#print("Tet mesh bounds:", tet_grid.bounds)
#print("Hex mesh bounds:", hex_grid.bounds)
#
## --- Plotting ---
#p = pv.Plotter()
#p.add_mesh(tet_grid, color="red", opacity=0.5, label="Tet Mesh")
#p.add_mesh(hex_grid, color="blue", opacity=0.3, label="Hex Mesh")
#p.add_axes()
#p.add_legend()
#p.show_grid()
#p.background_color = "white"
#p.show(title="Tet vs Hex Mesh Alignment")

#
#import numpy as np
#import meshio
#import SimpleITK as sitk
#
## --- Load mask ---
#mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"
#mask = sitk.ReadImage(mask_path)
#mask_arr = sitk.GetArrayFromImage(mask)  # shape (Z,Y,X)
#spacing = np.array(mask.GetSpacing())[::-1]  # z,y,x order
#origin = np.array(mask.GetOrigin())[::-1]
#
## --- Load tetrahedral mesh ---
#tet_mesh = meshio.read("tet_mesh.msh")  # or .vtu
#tets = tet_mesh.cells_dict["tetra"]
#points = tet_mesh.points
#
## --- Compute tetra centroids ---
#tet_centroids = points[tets].mean(axis=1)  # shape: (num_tets, 3)
#
## --- Map centroids to voxel indices ---
#voxel_indices = np.floor((tet_centroids - origin) / spacing).astype(int)
#
## Clip indices to valid range
#nz, ny, nx = mask_arr.shape
#voxel_indices[:,0] = np.clip(voxel_indices[:,0], 0, nz-1)
#voxel_indices[:,1] = np.clip(voxel_indices[:,1], 0, ny-1)
#voxel_indices[:,2] = np.clip(voxel_indices[:,2], 0, nx-1)
#
## --- Count tets per voxel ---
#counts = np.zeros(mask_arr.shape, dtype=int)
#for k, j, i in voxel_indices:
#    counts[k, j, i] += 1
#
## --- Example outputs ---
#print("Total voxels with at least one tet:", np.sum(counts>0))
#print("Max tets in a voxel:", counts.max())
#print("Min tets in a voxel (non-empty):", counts[counts>0].min())


#import numpy as np
#import meshio
#import SimpleITK as sitk
#import pyvista as pv
#
## --- Load meshes ---
#tet_file = "tet_mesh.vtu"
#hex_file = "hex_mesh.vtu"
#
#tet_mesh = meshio.read(tet_file)
#hex_mesh = meshio.read(hex_file)
#
## --- Load original mask for voxel grid info ---
#mask_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"
#mask = sitk.ReadImage(mask_path)
#mask_arr = sitk.GetArrayFromImage(mask)  # (Z,Y,X)
#spacing = mask.GetSpacing()
#origin = mask.GetOrigin()
#nz, ny, nx = mask_arr.shape
#
## --- Prepare voxel grid ---
#tet_counts = np.zeros((nz, ny, nx), dtype=int)
#
## --- Helper to compute voxel index for a point ---
#def point_to_voxel(pt):
#    # Convert physical coordinates to voxel indices
#    i = int((pt[0] - origin[0]) / spacing[0])
#    j = int((pt[1] - origin[1]) / spacing[1])
#    k = int((pt[2] - origin[2]) / spacing[2])
#    # Clip to grid
#    i = np.clip(i, 0, nx-1)
#    j = np.clip(j, 0, ny-1)
#    k = np.clip(k, 0, nz-1)
#    return k, j, i
#
## --- Count tets per voxel ---
#tets = tet_mesh.cells_dict["tetra"]
#points = tet_mesh.points
#
#for tet in tets:
#    tet_pts = points[tet]
#    # Use the centroid of the tetrahedron to assign voxel
#    centroid = tet_pts.mean(axis=0)
#    k,j,i = point_to_voxel(centroid)
#    tet_counts[k,j,i] += 1
#
## --- Convert voxel grid to PyVista UniformGrid ---
#grid = pv.UniformGrid()
#grid.dimensions = np.array(tet_counts.shape) + 1  # dimensions = nx+1, ny+1, nz+1
#grid.origin = origin
#grid.spacing = spacing
#grid.cell_data["tet_count"] = tet_counts.ravel(order="F")  # Fortran order for PyVista
#
## --- Plot ---
#p = pv.Plotter()
#p.add_volume(grid, scalars="tet_count", opacity="linear", cmap="viridis")
#p.add_axes()
#p.show_grid()
#p.show(title="Tets per Voxel Heatmap")



import numpy as np
import SimpleITK as sitk
import meshio
import subprocess
import pyvista as pv
from skimage import measure


# --- Helper functions ---

def mask_to_surface(mask_path, surface_file="surface.stl", decimate=0.2, smooth_iter=30):
    """Convert 3D mask/image to STL surface, clean, smooth, and decimate."""
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask)

    # marching cubes
    verts, faces, normals, values = measure.marching_cubes(arr, level=0.5)

    spacing = np.array(mask.GetSpacing())[::-1]  # z,y,x → match numpy
    origin = np.array(mask.GetOrigin())[::-1]
    points = verts * spacing + origin

    mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])
    tmp_file = surface_file.replace(".stl", "_raw.stl")
    meshio.write(tmp_file, mesh)

    # clean + smooth + decimate
    surf = pv.read(tmp_file).clean(tolerance=1e-6)
    surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.01)
    if decimate > 0:
        surf = surf.decimate(decimate)

    surf.save(surface_file)
    print(f"✅ Saved cleaned surface: {surface_file} (triangles: {surf.n_cells})")
    return surface_file


def write_geo(surface_file, geo_file="mesh.geo", lc=2.0):
    """Write Gmsh .geo script for tetrahedral mesh generation from STL."""
    geo_text = f"""
    Merge "{surface_file}";
    // Collect all surfaces from STL
    Surface Loop(1) = {{Surface[:]}};
    Volume(1) = {{1}};
    Physical Volume("domain") = {{1}};
    Mesh.Algorithm3D = 4;
    Mesh.CharacteristicLengthMax = {lc};
    """
    with open(geo_file, "w") as f:
        f.write(geo_text)
    return geo_file


def generate_tet_mesh(geo_file, out_file="mesh.msh", threads=8):
    """Run Gmsh to generate tetrahedral mesh."""
    cmd = ["gmsh", "-3", geo_file, "-format", "msh2", "-o", out_file, f"-nt{threads}"]
    print("🚀 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Generated tetrahedral mesh: {out_file}")
    return out_file


def load_mesh_pv(filename):
    """Load tetrahedral mesh as PyVista UnstructuredGrid."""
    mesh = meshio.read(filename)
    if "tetra" in mesh.cells_dict:
        return pv.UnstructuredGrid({10: mesh.cells_dict["tetra"]}, mesh.points)
    else:
        raise ValueError(f"No tetrahedral cells found in {filename}. Found: {mesh.cells_dict.keys()}")


# --- Paths ---
image_path = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz"
mask_path  = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz"

# --- Generate surface meshes ---
image_surface = mask_to_surface(image_path, "image_surface.stl", decimate=0.8, smooth_iter=20)
mask_surface  = mask_to_surface(mask_path, "mask_surface.stl", decimate=0.5, smooth_iter=20)

# --- Generate tetrahedral meshes ---
image_geo = write_geo(image_surface, "image.geo", lc=3.0)
mask_geo  = write_geo(mask_surface, "mask.geo", lc=2.0)

image_mesh_file = generate_tet_mesh(image_geo, "image_mesh.msh")
mask_mesh_file  = generate_tet_mesh(mask_geo, "mask_mesh.msh")

# --- Debug: check what cells exist ---
mesh = meshio.read(image_mesh_file)
print("Image mesh cells:", mesh.cells_dict.keys())
m = meshio.read(mask_mesh_file)
print("Mask mesh cells:", m.cells_dict.keys())

# --- Load meshes for visualization ---
image_mesh = load_mesh_pv(image_mesh_file)
mask_mesh  = load_mesh_pv(mask_mesh_file)

# --- Overlay in PyVista ---
p = pv.Plotter()
p.add_mesh(image_mesh, color="green", opacity=0.3, show_edges=False, label="MRI Image")
p.add_mesh(mask_mesh, color="red", opacity=0.6, show_edges=True, label="Tumor Mask")
p.add_axes()
p.add_legend()
p.show_grid()
p.show(title="MRI Image vs Tumor Mesh Overlay")

