import meshio
mesh = meshio.read("tumor_mesh_3d.msh")
mesh.write("tumor_mesh.vtu")
mesh.write("tumor_mesh_2.cgns")
