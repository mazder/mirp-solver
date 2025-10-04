import meshio
import numpy as np
import pyvista as pv

def load_mesh(mesh_file):
    mesh = meshio.read(mesh_file)
    points = mesh.points
    # Find tetrahedral elements
    cells = None
    for cell_block in mesh.cells:
        if cell_block.type in ["tetra", "tetra10"]:
            cells = cell_block.data
            break
    if cells is None:
        raise ValueError(f"No tetrahedral cells found in mesh {mesh_file}")
    return points, cells

def bounding_box(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs

def volume_tetrahedron(v0, v1, v2, v3):
    # Compute volume of a tetrahedron given 4 points
    return abs(np.dot(v3 - v0, np.cross(v1 - v0, v2 - v0))) / 6.0

def total_mesh_volume(points, tets):
    volumes = [volume_tetrahedron(points[t[0]], points[t[1]], points[t[2]], points[t[3]]) for t in tets]
    return np.sum(volumes)

def bounding_box_overlap(box1, box2):
    # Overlap between two bounding boxes (mins, maxs)
    mins1, maxs1 = box1
    mins2, maxs2 = box2
    overlap_min = np.maximum(mins1, mins2)
    overlap_max = np.minimum(maxs1, maxs2)
    overlap = np.maximum(overlap_max - overlap_min, 0)
    return np.prod(overlap)

def mesh_quality_stats(points, tets):
    # Simple metric: ratio of shortest edge to longest edge in each tet
    qualities = []
    for tet in tets:
        verts = points[tet]
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(verts[i] - verts[j]))
        ratio = min(edges) / max(edges) if max(edges) > 0 else 0
        qualities.append(ratio)
    qualities = np.array(qualities)
    return {
        "min_quality": qualities.min(),
        "max_quality": qualities.max(),
        "mean_quality": qualities.mean(),
        "std_quality": qualities.std(),
    }

def compare_meshes(mesh1_path, mesh2_path):
    print(f"Loading meshes:\n - {mesh1_path}\n - {mesh2_path}")
    points1, tets1 = load_mesh(mesh1_path)
    points2, tets2 = load_mesh(mesh2_path)

    print(f"\nMesh 1: {len(points1)} points, {len(tets1)} tetrahedra")
    print(f"Mesh 2: {len(points2)} points, {len(tets2)} tetrahedra")

    bb1 = bounding_box(points1)
    bb2 = bounding_box(points2)
    print("\nBounding boxes:")
    print(f"Mesh 1 mins: {bb1[0]}, maxs: {bb1[1]}")
    print(f"Mesh 2 mins: {bb2[0]}, maxs: {bb2[1]}")

    overlap_vol = bounding_box_overlap(bb1, bb2)
    print(f"Bounding box overlap volume: {overlap_vol}")

    vol1 = total_mesh_volume(points1, tets1)
    vol2 = total_mesh_volume(points2, tets2)
    print(f"\nApproximate total mesh volume:")
    print(f"Mesh 1 volume: {vol1:.2f}")
    print(f"Mesh 2 volume: {vol2:.2f}")
    print(f"Volume difference: {abs(vol1 - vol2):.2f} ({100 * abs(vol1 - vol2)/max(vol1, vol2):.2f}%)")

    q1 = mesh_quality_stats(points1, tets1)
    q2 = mesh_quality_stats(points2, tets2)
    print("\nMesh quality (edge length ratio):")
    print(f"Mesh 1 - min: {q1['min_quality']:.3f}, max: {q1['max_quality']:.3f}, mean: {q1['mean_quality']:.3f}, std: {q1['std_quality']:.3f}")
    print(f"Mesh 2 - min: {q2['min_quality']:.3f}, max: {q2['max_quality']:.3f}, mean: {q2['mean_quality']:.3f}, std: {q2['std_quality']:.3f}")

def visualize_meshes(mesh1_path, mesh2_path):
    mesh1 = pv.read(mesh1_path)
    mesh2 = pv.read(mesh2_path)

    p = pv.Plotter()
    p.add_mesh(mesh1, color="lightblue", opacity=0.3, label="Mesh 1")
    p.add_mesh(mesh2, color="orange", opacity=0.3, label="Mesh 2")
    p.add_legend()
    p.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two tetrahedral meshes.")
    parser.add_argument("mesh1", help="Path to first mesh file (VTU or MSH)")
    parser.add_argument("mesh2", help="Path to second mesh file (VTU or MSH)")
    parser.add_argument("--visualize", action="store_true", help="Visualize meshes overlay")

    args = parser.parse_args()

    compare_meshes(args.mesh1, args.mesh2)

    if args.visualize:
        visualize_meshes(args.mesh1, args.mesh2)
