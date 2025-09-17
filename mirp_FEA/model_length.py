import pyvista as pv

mesh = pv.read("tumor_surface_final.stl")
print(f"Bounds in mm: {mesh.bounds}")

bounds = mesh.bounds
size_m = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
print(f"Mesh size (in mm): X={size_m[0]:.4f}, Y={size_m[1]:.4f}, Z={size_m[2]:.4f}")


mesh = pv.read("tumer_mesh_3d.msh")
mesh.plot(show_edges=True)
