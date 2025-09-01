#Features in this single file:
#Reads .vtu tetrahedral mesh using meshio.
#Converts it into FEniCS mesh.
#Sets up a linear elasticity problem with Dirichlet BC and optional traction.
#Computes von Mises stress for all elements.
#Identifies top 5% stress voxels.
#Saves stress fields to XDMF for ParaView visualization.
#Optionally overlays hot stress on a CT mid-slice.

#Alternative: Docker (guaranteed working)
#docker pull quay.io/fenicsproject/stable:current
#docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:current
#Inside the container:
#cd /home/fenics/shared
#python3 fea_brain_tumor_from_vtu.py

# fea_brain_tumor_from_vtu.py

# fea_brain_tumor_from_vtu_dolfinx.py

import meshio
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD

# -------------------------------
# Step 0: Load VTU mesh
# -------------------------------
vtu_file = "EGD-0001_1_T1_0.vtu"
mesh_from_vtu = meshio.read(vtu_file)

points = mesh_from_vtu.points
cells = mesh_from_vtu.cells_dict["tetra"]

# Convert to Dolfinx mesh
# Note: Dolfinx prefers XDMF, but we can use MeshEditor
dolfin_mesh = mesh.Mesh(mesh.MeshTopology(), comm)
editor = mesh.MeshEditor()
editor.open(dolfin_mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(points))
editor.init_cells(len(cells))
for i, p in enumerate(points):
    editor.add_vertex(i, p)
for i, c in enumerate(cells):
    editor.add_cell(i, c)
editor.close()

# -------------------------------
# Step 1: Function spaces
# -------------------------------
V = fem.VectorFunctionSpace(dolfin_mesh, ("CG", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)

# -------------------------------
# Step 2: Material properties
# -------------------------------
E_brain, nu_brain = 3000.0, 0.45
E_tumor, nu_tumor = 5000.0, 0.45

mu_brain = E_brain / (2*(1+nu_brain))
lmbda_brain = E_brain*nu_brain / ((1+nu_brain)*(1-2*nu_brain))

# TODO: assign tumor region via markers
def sigma(u):
    return 2*mu_brain*ufl.sym(ufl.grad(u)) + lmbda_brain*ufl.tr(ufl.sym(ufl.grad(u)))*ufl.Identity(3)

# -------------------------------
# Step 3: BCs
# -------------------------------
# Fix bottom plane
def bottom_boundary(x):
    return np.isclose(x[2], 0.0)

bc_dofs = fem.locate_dofs_geometrical(V, bottom_boundary)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

# -------------------------------
# Step 4: Weak form (linear elasticity)
# -------------------------------
T = ufl.Constant(dolfin_mesh, PETSc.ScalarType((0,0,1.0)))
ds = ufl.Measure("ds", domain=dolfin_mesh)

a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v)))*ufl.dx
L = ufl.inner(T, v)*ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg"})

u_sol = problem.solve()

# -------------------------------
# Step 5: Compute von Mises stress
# -------------------------------
W = fem.TensorFunctionSpace(dolfin_mesh, ("DG", 0))
stress = fem.Function(W)
stress.interpolate(sigma(u_sol))

V0 = fem.FunctionSpace(dolfin_mesh, ("DG", 0))
d = u_sol.geometric_dimension()
s_dev = stress - (1./3)*ufl.tr(stress)*ufl.Identity(d)
von_mises = fem.Function(V0)
von_mises.interpolate(ufl.sqrt(3./2*ufl.inner(s_dev, s_dev)))

# -------------------------------
# Step 6: Top 5% stress
# -------------------------------
vm_vals = von_mises.x.array  # Dolfinx uses PETSc vectors
threshold = np.percentile(vm_vals, 95)

hot_voxels = np.copy(vm_vals)
hot_voxels[hot_voxels < threshold] = 0.0
von_mises_hot = fem.Function(V0)
von_mises_hot.x.array[:] = hot_voxels

# -------------------------------
# Step 7: Save results
# -------------------------------
with io.XDMFFile(comm, "von_mises_full.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfin_mesh)
    xdmf.write_function(von_mises)

with io.XDMFFile(comm, "von_mises_hot5.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfin_mesh)
    xdmf.write_function(von_mises_hot)

print("✅ Stress fields saved")

# -------------------------------
# Step 8: Optional overlay on CT mid-slice
# -------------------------------
ct_img = nib.load("patient_ct.nii.gz")
ct_arr = ct_img.get_fdata()
mid_slice = ct_arr.shape[2] // 2

plt.figure(figsize=(8,8))
plt.imshow(ct_arr[:,:,mid_slice].T, cmap='gray', origin='lower')

# crude projection of stress onto slice (approximate)
hot_arr = np.zeros_like(ct_arr)
hot_arr[:,:,mid_slice] = hot_voxels[:ct_arr.shape[0]*ct_arr.shape[1]].reshape(ct_arr.shape[0], ct_arr.shape[1])
plt.imshow(hot_arr[:,:,mid_slice].T, cmap='hot', alpha=0.5, origin='lower')

plt.title("CT mid slice with top 5% stress overlay")
plt.axis('off')
plt.savefig("ct_overlay_hot5.png", dpi=300)
print("✅ Overlay saved: ct_overlay_hot5.png")

