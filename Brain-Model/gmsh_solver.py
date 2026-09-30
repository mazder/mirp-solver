#!/usr/bin/env python3
"""
gmsh_solver.py

MRI-ONLY 5-REGION CONFORMING VOXEL FEM MESH GENERATOR
======================================================

Regions:
    1 = BrainTissue
    2 = Tumor
    3 = CSF
    4 = Skull
    5 = Scalp

The solver consumes the DERIVED FEM masks produced by preprocess.py.

Important:
    - Original MRI masks are read-only.
    - Original tumor segmentation is read-only and must match exactly.
    - No additional padding is performed here.
    - No morphology is performed here.
    - No marching cubes is used for the FEM mesh.
    - Every occupied voxel becomes one HEX8 element.
    - Material interfaces and the Scalp exterior are QUAD4 elements.
    - Only these material interfaces are allowed:
          Tumor <-> BrainTissue
          BrainTissue <-> CSF
          CSF <-> Skull
          Skull <-> Scalp
    - Only Scalp may touch Outside.

Output:
    results/head_tumor_multiregion_mesh/
        head_tumor_5region.msh
        material_labels.nii.gz
        mesh_summary.json
        interface_counts.csv
        region_counts.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR

DATASET_DIR = MODEL_ROOT / "EGD-0001" / "MR_EGD-0001"
DATASET_MASK_DIR = DATASET_DIR / "preprocess" / "masks"
ROOT_MASK_DIR = MODEL_ROOT / "preprocess" / "masks"
MASK_DIR = DATASET_MASK_DIR if DATASET_MASK_DIR.exists() else ROOT_MASK_DIR

BRAIN_MASK_FILE = MASK_DIR / "brain_mask.nii.gz"
TUMOR_MASK_FILE = MASK_DIR / "tumor_mask.nii.gz"
CSF_MASK_FILE = MASK_DIR / "csf_mask.nii.gz"
SKULL_MASK_FILE = MASK_DIR / "skull_mask.nii.gz"
SCALP_MASK_FILE = MASK_DIR / "scalp_mask.nii.gz"

# IMPORTANT: module-level constant. Do NOT redefine this inside main().
MATERIAL_LABEL_FILE = MASK_DIR / "material_labels.nii.gz"

RAW_TUMOR_SOURCE = DATASET_DIR / "5_MASK" / "NIFTI" / "MASK.nii.gz"
DATASET_PREPROCESS_REPORT = DATASET_DIR / "preprocess" / "preprocess_report.json"
ROOT_PREPROCESS_REPORT = MODEL_ROOT / "preprocess" / "preprocess_report.json"
PREPROCESS_REPORT = (DATASET_PREPROCESS_REPORT if DATASET_PREPROCESS_REPORT.exists()
                     else ROOT_PREPROCESS_REPORT)

RESULT_DIR = DATASET_DIR / "results" / "head_tumor_multiregion_mesh"

MESH_FILE = RESULT_DIR / "head_tumor_5region.msh"
MATERIAL_LABELS_OUTPUT = RESULT_DIR / "material_labels.nii.gz"
MESH_SUMMARY_FILE = RESULT_DIR / "mesh_summary.json"
MESH_REPORT_FILE = RESULT_DIR / "mesh_report.json"
REGION_TAGS_FILE = RESULT_DIR / "region_tags.json"
INTERFACE_CSV = RESULT_DIR / "interface_counts.csv"
REGION_CSV = RESULT_DIR / "region_counts.csv"


# ============================================================================
# MATERIAL / PHYSICAL IDS
# ============================================================================

MATERIAL_IDS = {
    "brain": 1,
    "tumor": 2,
    "csf": 3,
    "skull": 4,
    "scalp": 5,
}

MATERIAL_NAMES = {
    0: "Outside",
    1: "BrainTissue",
    2: "Tumor",
    3: "CSF",
    4: "Skull",
    5: "Scalp",
}

SURFACE_IDS = {
    "tumor_interface": 101,
    "brain_csf_interface": 102,
    "csf_skull_interface": 103,
    "skull_scalp_interface": 104,
    "scalp_exterior": 105,
}

SURFACE_NAMES = {
    101: "TumorInterface",
    102: "BrainCSFInterface",
    103: "CSFSkullInterface",
    104: "SkullScalpInterface",
    105: "ScalpExterior",
}

INTERFACE_DEFINITIONS = [
    (1, 2, 101, "TumorInterface"),
    (1, 3, 102, "BrainCSFInterface"),
    (3, 4, 103, "CSFSkullInterface"),
    (4, 5, 104, "SkullScalpInterface"),
]

ALLOWED_INTERFACE_PAIRS = {
    tuple(sorted((a, b)))
    for a, b, _, _ in INTERFACE_DEFINITIONS
}

# Inner -> outer hierarchy.
MATERIAL_RANK = {
    2: 0,  # Tumor
    1: 1,  # Brain
    3: 2,  # CSF
    4: 3,  # Skull
    5: 4,  # Scalp
}

GMSH_QUAD4 = 3
GMSH_HEX8 = 5


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str = "") -> None:
    print(f"[gmsh_solver] {message}", flush=True)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def fatal(message: str) -> None:
    log("")
    log("FATAL ERROR")
    log(message)
    sys.exit(1)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self.start
        log(f"{self.name}: {elapsed:.2f} seconds")
        return elapsed


# ============================================================================
# INPUT / OUTPUT HELPERS
# ============================================================================

def prepare_directories() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def check_required_files() -> None:
    section("CHECKING INPUT FILES")

    required = {
        "brain mask": BRAIN_MASK_FILE,
        "tumor mask": TUMOR_MASK_FILE,
        "CSF mask": CSF_MASK_FILE,
        "skull mask": SKULL_MASK_FILE,
        "scalp mask": SCALP_MASK_FILE,
        "material labels": MATERIAL_LABEL_FILE,
        "original tumor": RAW_TUMOR_SOURCE,
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {name}:\n{path}")
        log(f"{name:<18}: {path}")

    log("All required input files exist.")


def load_binary_mask(path: Path, name: str):
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj)

    mask = np.asarray(
        np.isfinite(data) & (data > 0),
        dtype=bool,
    )

    log(
        f"{name:<12}: shape={mask.shape}, "
        f"voxels={int(mask.sum()):,}"
    )

    return mask, nii.affine.copy(), nii.header.copy()


def load_material_labels():
    log(f"Loading material labels: {MATERIAL_LABEL_FILE}")

    nii = nib.load(str(MATERIAL_LABEL_FILE))
    data = np.asarray(nii.dataobj)

    if not np.all(np.isfinite(data)):
        raise RuntimeError("material_labels.nii.gz contains non-finite values.")

    rounded = np.rint(data)

    if not np.array_equal(data, rounded):
        raise RuntimeError(
            "material_labels.nii.gz contains non-integer labels."
        )

    material = rounded.astype(np.uint8)

    unique = np.unique(material)
    invalid = [int(x) for x in unique if int(x) not in range(6)]

    if invalid:
        raise RuntimeError(f"Invalid material labels found: {invalid}")

    for mid in range(6):
        log(
            f"  {MATERIAL_NAMES[mid]:<15}: "
            f"{int(np.count_nonzero(material == mid)):,}"
        )

    return material, nii.affine.copy(), nii.header.copy()


# ============================================================================
# GEOMETRY VALIDATION
# ============================================================================

def voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise RuntimeError(f"Invalid voxel spacing: {spacing}")

    return spacing


def validate_mask_geometry(
    masks: dict[str, np.ndarray],
    affines: dict[str, np.ndarray],
    reference_affine: np.ndarray,
) -> None:
    section("VALIDATING MASK GEOMETRY")

    names = list(masks)
    reference_shape = masks[names[0]].shape

    for name in names:
        if masks[name].shape != reference_shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: {masks[name].shape}; "
                f"reference={reference_shape}"
            )

        if not np.allclose(
            affines[name],
            reference_affine,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(f"Affine mismatch for {name}.")

    log(f"Reference shape: {reference_shape}")
    log("All five material masks have identical geometry.")


def validate_material_geometry(
    material: np.ndarray,
    material_affine: np.ndarray,
    reference_shape: tuple[int, ...],
    reference_affine: np.ndarray,
) -> None:
    if material.shape != reference_shape:
        raise RuntimeError(
            f"material_labels shape mismatch: "
            f"{material.shape} != {reference_shape}"
        )

    if not np.allclose(
        material_affine,
        reference_affine,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "material_labels affine does not match material masks."
        )

    log("material_labels geometry matches masks.")


def validate_mask_exclusivity(masks: dict[str, np.ndarray]) -> None:
    section("CHECKING MATERIAL-MASK MUTUAL EXCLUSIVITY")

    names = list(masks)

    for a_idx in range(len(names)):
        for b_idx in range(a_idx + 1, len(names)):
            a_name = names[a_idx]
            b_name = names[b_idx]
            count = int(np.count_nonzero(masks[a_name] & masks[b_name]))

            if count:
                raise RuntimeError(
                    f"Material masks overlap: "
                    f"{a_name} ∩ {b_name} = {count:,}"
                )

    log("All five material masks are mutually exclusive.")


def build_material_from_masks(masks: dict[str, np.ndarray]) -> np.ndarray:
    material = np.zeros(masks["brain"].shape, dtype=np.uint8)

    for name, mid in MATERIAL_IDS.items():
        material[masks[name]] = mid

    return material


def validate_material_against_masks(
    material: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    section("VERIFYING MATERIAL LABELS AGAINST FIVE MASKS")

    reconstructed = build_material_from_masks(masks)

    if not np.array_equal(material, reconstructed):
        diff = int(np.count_nonzero(material != reconstructed))
        raise RuntimeError(
            "material_labels.nii.gz does not match the five material masks. "
            f"Differing voxels: {diff:,}"
        )

    log("material_labels.nii.gz exactly matches the five material masks.")


# ============================================================================
# EXACT TUMOR PRESERVATION
# ============================================================================

def verify_tumor_exact(
    material: np.ndarray,
    derived_affine: np.ndarray,
) -> dict:
    section("EXACT TUMOR COMPARISON AGAINST ORIGINAL SOURCE")

    source_nii = nib.load(str(RAW_TUMOR_SOURCE))
    source_data = np.asarray(source_nii.dataobj)

    source_tumor = np.asarray(
        np.isfinite(source_data) & (source_data > 0),
        dtype=bool,
    )

    derived_tumor = material == 2

    source_count = int(source_tumor.sum())
    derived_count = int(derived_tumor.sum())

    log(f"Original tumor shape: {source_tumor.shape}")
    log(f"Derived tumor shape:  {derived_tumor.shape}")
    log(f"Original tumor voxels: {source_count:,}")
    log(f"Derived tumor voxels:  {derived_count:,}")

    if source_count != derived_count:
        raise RuntimeError(
            "Tumor voxel count changed: "
            f"source={source_count:,}, derived={derived_count:,}"
        )

    source_origin = source_nii.affine[:3, 3]
    derived_origin = derived_affine[:3, 3]
    derived_linear = derived_affine[:3, :3]

    try:
        offset_float = np.linalg.solve(
            derived_linear,
            source_origin - derived_origin,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Derived affine is singular.") from exc

    offset = np.rint(offset_float).astype(np.int64)

    if not np.allclose(offset_float, offset, atol=1e-4, rtol=0):
        raise RuntimeError(
            "Could not determine an integer source-to-derived offset: "
            f"{offset_float}"
        )

    ox, oy, oz = map(int, offset)
    sx, sy, sz = source_tumor.shape
    dx, dy, dz = derived_tumor.shape

    log(f"Detected source offset in derived grid: ({ox}, {oy}, {oz})")

    if (
        ox < 0 or oy < 0 or oz < 0
        or ox + sx > dx
        or oy + sy > dy
        or oz + sz > dz
    ):
        raise RuntimeError("Original tumor does not fit inside derived domain.")

    extracted = derived_tumor[
        ox:ox + sx,
        oy:oy + sy,
        oz:oz + sz,
    ]

    equal = np.array_equal(extracted, source_tumor)
    difference_count = int(np.count_nonzero(extracted != source_tumor))

    log(f"Differing voxels:     {difference_count:,}")
    log(f"Exact tumor match:    {equal}")

    if not equal:
        raise RuntimeError(
            "Derived tumor mask does not exactly match "
            "the original tumor segmentation."
        )

    log("Tumor preservation check PASSED.")

    return {
        "exact_match": True,
        "source_voxels": source_count,
        "derived_voxels": derived_count,
        "difference_voxels": difference_count,
        "derived_offset": [ox, oy, oz],
    }


# ============================================================================
# MATERIAL INTERFACE VALIDATION
# ============================================================================

def validate_material_interfaces(material: np.ndarray) -> dict[int, dict]:
    section("VALIDATING MATERIAL INTERFACES")

    pair_counts: dict[tuple[int, int], int] = {}
    unexpected: set[tuple[int, int]] = set()

    def process(a: np.ndarray, b: np.ndarray) -> None:
        diff = (a > 0) & (b > 0) & (a != b)

        if not np.any(diff):
            return

        aa = a[diff].astype(np.int16)
        bb = b[diff].astype(np.int16)

        pairs = np.column_stack((np.minimum(aa, bb), np.maximum(aa, bb)))
        unique, counts = np.unique(pairs, axis=0, return_counts=True)

        for pair, count in zip(unique, counts):
            key = (int(pair[0]), int(pair[1]))
            pair_counts[key] = pair_counts.get(key, 0) + int(count)

            if key not in ALLOWED_INTERFACE_PAIRS:
                unexpected.add(key)

    process(material[:-1, :, :], material[1:, :, :])
    process(material[:, :-1, :], material[:, 1:, :])
    process(material[:, :, :-1], material[:, :, 1:])

    if unexpected:
        details = []
        for a, b in sorted(unexpected):
            details.append(
                f"{MATERIAL_NAMES[a]}({a}) <-> "
                f"{MATERIAL_NAMES[b]}({b}) = "
                f"{pair_counts[(a, b)]:,} faces"
            )
        raise RuntimeError(
            "Unexpected material interfaces detected:\n"
            + "\n".join(details)
        )

    result = {}

    for m1, m2, tag, name in INTERFACE_DEFINITIONS:
        pair = tuple(sorted((m1, m2)))
        count = int(pair_counts.get(pair, 0))

        result[tag] = {
            "name": name,
            "material_a": MATERIAL_NAMES[m1],
            "material_b": MATERIAL_NAMES[m2],
            "count": count,
        }

        log(f"{name:<25}: {count:,} faces")

    log("All direct material interfaces are valid.")
    return result


def count_exterior_faces(material: np.ndarray) -> dict[int, int]:
    section("VALIDATING MATERIAL-TO-OUTSIDE INTERFACES")

    padded = np.pad(material, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1, 1:-1]

    neighbors = [
        padded[:-2, 1:-1, 1:-1],
        padded[2:, 1:-1, 1:-1],
        padded[1:-1, :-2, 1:-1],
        padded[1:-1, 2:, 1:-1],
        padded[1:-1, 1:-1, :-2],
        padded[1:-1, 1:-1, 2:],
    ]

    counts = {mid: 0 for mid in range(1, 6)}

    for neighbor in neighbors:
        boundary = (center > 0) & (neighbor == 0)
        if np.any(boundary):
            values = center[boundary]
            unique, n = np.unique(values, return_counts=True)
            for mid, count in zip(unique, n):
                counts[int(mid)] += int(count)

    for mid in range(1, 6):
        log(
            f"{MATERIAL_NAMES[mid]:<15} -> Outside: "
            f"{counts[mid]:,}"
        )

    for mid in range(1, 5):
        if counts[mid] != 0:
            raise RuntimeError(
                f"Invalid exterior interface: "
                f"{MATERIAL_NAMES[mid]} touches Outside."
            )

    if counts[5] == 0:
        raise RuntimeError("Scalp does not touch Outside.")

    log("Exterior topology validation PASSED.")
    return counts


# ============================================================================
# CONFORMING NODE GRID
# ============================================================================

def build_node_index(material: np.ndarray):
    section("BUILDING CONFORMING FEM NODE GRID")

    nx, ny, nz = material.shape
    occupied = material > 0

    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)

    used[:-1, :-1, :-1] |= occupied
    used[1:, :-1, :-1] |= occupied
    used[:-1, 1:, :-1] |= occupied
    used[1:, 1:, :-1] |= occupied
    used[:-1, :-1, 1:] |= occupied
    used[1:, :-1, 1:] |= occupied
    used[:-1, 1:, 1:] |= occupied
    used[1:, 1:, 1:] |= occupied

    node_count = int(used.sum())

    node_index = np.full(
        used.shape,
        -1,
        dtype=np.int64,
    )
    node_index[used] = np.arange(
        1,
        node_count + 1,
        dtype=np.int64,
    )

    log(f"Voxel grid: {nx} x {ny} x {nz}")
    log(f"Node grid:  {nx + 1} x {ny + 1} x {nz + 1}")
    log(f"Used FEM nodes: {node_count:,}")

    return used, node_index, node_count


# ============================================================================
# CONNECTIVITY
# ============================================================================

def hex_connectivity(
    i: np.ndarray,
    j: np.ndarray,
    k: int,
    node_index: np.ndarray,
    positive_orientation: bool,
) -> np.ndarray:
    n0 = node_index[i,     j,     k]
    n1 = node_index[i + 1, j,     k]
    n2 = node_index[i + 1, j + 1, k]
    n3 = node_index[i,     j + 1, k]
    n4 = node_index[i,     j,     k + 1]
    n5 = node_index[i + 1, j,     k + 1]
    n6 = node_index[i + 1, j + 1, k + 1]
    n7 = node_index[i,     j + 1, k + 1]

    conn = np.column_stack((n0, n1, n2, n3, n4, n5, n6, n7))

    if not positive_orientation:
        conn = conn[:, [0, 3, 2, 1, 4, 7, 6, 5]]

    return conn.astype(np.int64, copy=False)


def quad_face(
    side: str,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    """
    Return QUAD4 nodes with outward normal corresponding to `side`.

    Sides:
        xm = -X
        xp = +X
        ym = -Y
        yp = +Y
        zm = -Z
        zp = +Z
    """
    if side == "xm":
        nodes = (
            node_index[i, j, k],
            node_index[i, j, k + 1],
            node_index[i, j + 1, k + 1],
            node_index[i, j + 1, k],
        )
    elif side == "xp":
        nodes = (
            node_index[i + 1, j, k],
            node_index[i + 1, j + 1, k],
            node_index[i + 1, j + 1, k + 1],
            node_index[i + 1, j, k + 1],
        )
    elif side == "ym":
        nodes = (
            node_index[i, j, k],
            node_index[i + 1, j, k],
            node_index[i + 1, j, k + 1],
            node_index[i, j, k + 1],
        )
    elif side == "yp":
        nodes = (
            node_index[i, j + 1, k],
            node_index[i, j + 1, k + 1],
            node_index[i + 1, j + 1, k + 1],
            node_index[i + 1, j + 1, k],
        )
    elif side == "zm":
        nodes = (
            node_index[i, j, k],
            node_index[i, j + 1, k],
            node_index[i + 1, j + 1, k],
            node_index[i + 1, j, k],
        )
    elif side == "zp":
        nodes = (
            node_index[i, j, k + 1],
            node_index[i + 1, j, k + 1],
            node_index[i + 1, j + 1, k + 1],
            node_index[i, j + 1, k + 1],
        )
    else:
        raise ValueError(f"Unknown face side: {side}")

    return np.column_stack(nodes).astype(np.int64, copy=False)


def write_quad_records(
    f,
    element_ids: np.ndarray,
    conn: np.ndarray,
    physical_tag: int,
) -> None:
    if len(element_ids) == 0:
        return

    rows = np.column_stack(
        (
            element_ids,
            np.full(len(element_ids), GMSH_QUAD4, dtype=np.int64),
            np.full(len(element_ids), 2, dtype=np.int64),
            np.full(len(element_ids), physical_tag, dtype=np.int64),
            np.full(len(element_ids), physical_tag, dtype=np.int64),
            conn,
        )
    )

    np.savetxt(f, rows, fmt="%d")


# ============================================================================
# INTERFACE FACE ITERATION
# ============================================================================

def interface_faces(material: np.ndarray, m1: int, m2: int):
    """
    Yield (side, i, j, k) for each interface face.

    The returned side belongs to the inner material according to
    MATERIAL_RANK, so its outward normal points inner -> outer.
    """
    rank1 = MATERIAL_RANK[m1]
    rank2 = MATERIAL_RANK[m2]

    # X direction: voxel (i,j,k) and (i+1,j,k)
    a = material[:-1, :, :]
    b = material[1:, :, :]

    for inner, outer in ((m1, m2), (m2, m1)):
        mask = (a == inner) & (b == outer)
        i, j, k = np.nonzero(mask)
        if len(i):
            side = "xp" if rank1 < rank2 else "xm"
            if inner == m2:
                side = "xm" if rank1 < rank2 else "xp"
            # More directly determine side from which voxel is inner.
            side = "xp" if inner == m1 else "xm"
            if MATERIAL_RANK[inner] > MATERIAL_RANK[outer]:
                side = "xp" if inner == m1 else "xm"
            yield side, i, j, k

    # Y direction
    a = material[:, :-1, :]
    b = material[:, 1:, :]

    for inner in (m1, m2):
        outer = m2 if inner == m1 else m1
        mask = (a == inner) & (b == outer)
        i, j, k = np.nonzero(mask)
        if len(i):
            side = "yp" if inner == m1 else "ym"
            if MATERIAL_RANK[inner] > MATERIAL_RANK[outer]:
                side = "yp" if inner == m1 else "ym"
            yield side, i, j, k

    # Z direction
    a = material[:, :, :-1]
    b = material[:, :, 1:]

    for inner in (m1, m2):
        outer = m2 if inner == m1 else m1
        mask = (a == inner) & (b == outer)
        i, j, k = np.nonzero(mask)
        if len(i):
            side = "zp" if inner == m1 else "zm"
            if MATERIAL_RANK[inner] > MATERIAL_RANK[outer]:
                side = "zp" if inner == m1 else "zm"
            yield side, i, j, k


def write_internal_interfaces(
    f,
    material: np.ndarray,
    node_index: np.ndarray,
    element_id: int,
) -> tuple[int, dict[int, int]]:
    section("WRITING MATERIAL INTERFACE QUAD4 ELEMENTS")

    written_counts = {}

    for m1, m2, tag, name in INTERFACE_DEFINITIONS:
        before = element_id
        count = 0

        for side, i, j, k in interface_faces(material, m1, m2):
            conn = quad_face(side, i, j, k, node_index)
            ids = np.arange(
                element_id,
                element_id + len(i),
                dtype=np.int64,
            )
            write_quad_records(f, ids, conn, tag)
            element_id += len(i)
            count += len(i)

        written_counts[tag] = count

        log(f"{name:<25}: {count:,} QUAD4")

        if count != count_pair_faces(material, m1, m2):
            raise RuntimeError(
                f"Interface write count mismatch for {name}: "
                f"written={count}, expected={count_pair_faces(material, m1, m2)}"
            )

    return element_id, written_counts


def count_pair_faces(material: np.ndarray, m1: int, m2: int) -> int:
    count = 0

    count += int(np.count_nonzero(
        ((material[:-1, :, :] == m1) & (material[1:, :, :] == m2))
        | ((material[:-1, :, :] == m2) & (material[1:, :, :] == m1))
    ))

    count += int(np.count_nonzero(
        ((material[:, :-1, :] == m1) & (material[:, 1:, :] == m2))
        | ((material[:, :-1, :] == m2) & (material[:, 1:, :] == m1))
    ))

    count += int(np.count_nonzero(
        ((material[:, :, :-1] == m1) & (material[:, :, 1:] == m2))
        | ((material[:, :, :-1] == m2) & (material[:, :, 1:] == m1))
    ))

    return count


# ============================================================================
# SCALP EXTERIOR FACES
# ============================================================================

def write_scalp_exterior_faces(
    f,
    material: np.ndarray,
    node_index: np.ndarray,
    element_id: int,
    physical_tag: int,
) -> int:
    section("WRITING SCALP EXTERIOR QUAD4 ELEMENTS")

    start = element_id

    nx, ny, nz = material.shape

    # Internal zero-neighbor faces.
    tests = [
        (
            "xm",
            material[1:, :, :] == 0,
            material[:-1, :, :] == 5,
            lambda i, j, k: (i, j, k),
        ),
        (
            "xp",
            material[:-1, :, :] == 0,
            material[1:, :, :] == 5,
            lambda i, j, k: (i + 1, j, k),
        ),
        (
            "ym",
            material[:, 1:, :] == 0,
            material[:, :-1, :] == 5,
            lambda i, j, k: (i, j, k),
        ),
        (
            "yp",
            material[:, :-1, :] == 0,
            material[:, 1:, :] == 5,
            lambda i, j, k: (i, j + 1, k),
        ),
        (
            "zm",
            material[:, :, 1:] == 0,
            material[:, :, :-1] == 5,
            lambda i, j, k: (i, j, k),
        ),
        (
            "zp",
            material[:, :, :-1] == 0,
            material[:, :, 1:] == 5,
            lambda i, j, k: (i, j, k + 1),
        ),
    ]

    for side, zero_side, scalp_side, index_fn in tests:
        mask = zero_side & scalp_side
        i, j, k = np.nonzero(mask)

        if len(i) == 0:
            continue

        i2, j2, k2 = index_fn(i, j, k)
        conn = quad_face(side, i2, j2, k2, node_index)

        ids = np.arange(
            element_id,
            element_id + len(i),
            dtype=np.int64,
        )
        write_quad_records(f, ids, conn, physical_tag)
        element_id += len(i)

    # Six array boundaries: outside is implicit beyond the derived domain.
    boundary_tests = [
        ("xm", material[0, :, :] == 5, lambda j, k: (
            np.zeros(len(j), dtype=np.int64), j, k
        )),
        ("xp", material[-1, :, :] == 5, lambda j, k: (
            np.full(len(j), nx - 1, dtype=np.int64), j, k
        )),
        ("ym", material[:, 0, :] == 5, lambda i, k: (
            i, np.zeros(len(i), dtype=np.int64), k
        )),
        ("yp", material[:, -1, :] == 5, lambda i, k: (
            i, np.full(len(i), ny - 1, dtype=np.int64), k
        )),
        ("zm", material[:, :, 0] == 5, lambda i, j: (
            i, j, np.zeros(len(i), dtype=np.int64)
        )),
        ("zp", material[:, :, -1] == 5, lambda i, j: (
            i, j, np.full(len(i), nz - 1, dtype=np.int64)
        )),
    ]

    for side, mask, index_fn in boundary_tests:
        a, b = np.nonzero(mask)
        if len(a) == 0:
            continue

        i, j, k = index_fn(a, b)
        conn = quad_face(side, i, j, k, node_index)

        ids = np.arange(
            element_id,
            element_id + len(a),
            dtype=np.int64,
        )
        write_quad_records(f, ids, conn, physical_tag)
        element_id += len(a)

    count = element_id - start
    log(f"Scalp exterior QUAD4 faces: {count:,}")

    return element_id


def count_scalp_exterior_faces(material: np.ndarray) -> int:
    padded = np.pad(material, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1, 1:-1]

    count = 0
    for neighbor in (
        padded[:-2, 1:-1, 1:-1],
        padded[2:, 1:-1, 1:-1],
        padded[1:-1, :-2, 1:-1],
        padded[1:-1, 2:, 1:-1],
        padded[1:-1, 1:-1, :-2],
        padded[1:-1, 1:-1, 2:],
    ):
        count += int(np.count_nonzero((center == 5) & (neighbor == 0)))

    return count


# ============================================================================
# NODE WRITING
# ============================================================================

def write_nodes(
    f,
    used: np.ndarray,
    node_index: np.ndarray,
    affine: np.ndarray,
    node_count: int,
) -> None:
    f.write("$Nodes\n")
    f.write(f"{node_count}\n")

    linear = affine[:3, :3]
    origin = affine[:3, 3]

    for k in range(used.shape[2]):
        i, j = np.nonzero(used[:, :, k])

        if len(i) == 0:
            continue

        ids = node_index[i, j, k]

        voxel_xyz = np.column_stack(
            (
                i.astype(np.float64),
                j.astype(np.float64),
                np.full(len(i), float(k)),
            )
        )

        world_xyz = voxel_xyz @ linear.T + origin

        for node_id, xyz in zip(ids, world_xyz):
            f.write(
                f"{int(node_id)} "
                f"{xyz[0]:.12g} "
                f"{xyz[1]:.12g} "
                f"{xyz[2]:.12g}\n"
            )

    f.write("$EndNodes\n")


# ============================================================================
# VOLUME ELEMENT WRITING
# ============================================================================

def write_volume_elements(
    f,
    material: np.ndarray,
    node_index: np.ndarray,
    first_element_id: int,
    affine: np.ndarray,
) -> tuple[int, dict[int, int]]:
    section("WRITING HEX8 VOLUME ELEMENTS")

    determinant = float(np.linalg.det(affine[:3, :3]))

    if abs(determinant) < 1e-12:
        raise RuntimeError("Affine determinant is zero.")

    positive_orientation = determinant > 0

    total = int(np.count_nonzero(material > 0))
    log(f"Affine determinant: {determinant:.12g}")
    log(
        "HEX8 orientation: "
        + ("standard" if positive_orientation else "reversed")
    )
    log(f"Expected volume elements: {total:,}")

    element_id = first_element_id
    region_counts = {mid: 0 for mid in range(1, 6)}

    nz = material.shape[2]

    for k in range(nz):
        occupied = material[:, :, k] > 0
        if not np.any(occupied):
            continue

        i, j = np.nonzero(occupied)

        conn = hex_connectivity(
            i,
            j,
            k,
            node_index,
            positive_orientation,
        )

        labels = material[i, j, k].astype(np.int64, copy=False)

        ids = np.arange(
            element_id,
            element_id + len(i),
            dtype=np.int64,
        )

        rows = np.column_stack(
            (
                ids,
                np.full(len(i), GMSH_HEX8, dtype=np.int64),
                np.full(len(i), 2, dtype=np.int64),
                labels,
                labels,
                conn,
            )
        )

        np.savetxt(f, rows, fmt="%d")

        element_id += len(i)

        unique, counts = np.unique(labels, return_counts=True)
        for mid, count in zip(unique, counts):
            region_counts[int(mid)] += int(count)

        if k % 10 == 0 or k == nz - 1:
            log(
                f"  z-slice {k + 1}/{nz}: "
                f"{element_id - first_element_id:,}/{total:,}"
            )

    written = element_id - first_element_id

    if written != total:
        raise RuntimeError(
            f"Volume element count mismatch: "
            f"expected={total:,}, written={written:,}"
        )

    return element_id, region_counts


# ============================================================================
# PHYSICAL NAMES / CSV / OUTPUT
# ============================================================================

def write_physical_names(f) -> None:
    entries = [
        (3, 1, "BrainTissue"),
        (3, 2, "Tumor"),
        (3, 3, "CSF"),
        (3, 4, "Skull"),
        (3, 5, "Scalp"),
        (2, 101, "TumorInterface"),
        (2, 102, "BrainCSFInterface"),
        (2, 103, "CSFSkullInterface"),
        (2, 104, "SkullScalpInterface"),
        (2, 105, "ScalpExterior"),
    ]

    f.write("$PhysicalNames\n")
    f.write(f"{len(entries)}\n")

    for dim, tag, name in entries:
        f.write(f'{dim} {tag} "{name}"\n')

    f.write("$EndPhysicalNames\n")


def write_region_csv(region_counts: dict[int, int]) -> None:
    with REGION_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material_id", "material", "hex8_elements"])

        for mid in range(1, 6):
            writer.writerow(
                [mid, MATERIAL_NAMES[mid], region_counts[mid]]
            )


def write_interface_csv(interface_counts: dict[int, int]) -> None:
    with INTERFACE_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["physical_id", "surface", "quad4_elements"])

        for tag in sorted(interface_counts):
            writer.writerow(
                [tag, SURFACE_NAMES[tag], interface_counts[tag]]
            )


def write_material_output(
    material: np.ndarray,
    affine: np.ndarray,
    reference_header,
) -> None:
    image = nib.Nifti1Image(
        material.astype(np.uint8),
        affine,
        reference_header,
    )
    image.set_data_dtype(np.uint8)
    nib.save(image, str(MATERIAL_LABELS_OUTPUT))

    log(f"FEM material labels written to: {MATERIAL_LABELS_OUTPUT}")


def load_preprocess_report():
    if not PREPROCESS_REPORT.exists():
        return None

    try:
        with PREPROCESS_REPORT.open("r") as f:
            return json.load(f)
    except Exception as exc:
        log(f"WARNING: could not read preprocess report: {exc}")
        return None


# ============================================================================
# GMSH MESH WRITING
# ============================================================================

def write_gmsh_mesh(
    material: np.ndarray,
    used: np.ndarray,
    node_index: np.ndarray,
    node_count: int,
    affine: np.ndarray,
    interface_counts: dict[int, int],
) -> dict:
    section("WRITING GMSH MESH")

    volume_count = int(np.count_nonzero(material > 0))
    surface_count = int(sum(interface_counts.values()))
    total_count = volume_count + surface_count

    log(f"Nodes:          {node_count:,}")
    log(f"HEX8 volume:    {volume_count:,}")
    log(f"QUAD4 surfaces: {surface_count:,}")
    log(f"Total elements: {total_count:,}")

    surface_written: dict[int, int] = {}

    with MESH_FILE.open(
        "w",
        buffering=1024 * 1024,
    ) as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        write_physical_names(f)
        write_nodes(f, used, node_index, affine, node_count)

        f.write("$Elements\n")
        f.write(f"{total_count}\n")

        element_id = 1

        # Surfaces first. The total element count is known before writing.
        element_id, internal_written = write_internal_interfaces(
            f,
            material,
            node_index,
            element_id,
        )

        surface_written.update(internal_written)

        before = element_id
        element_id = write_scalp_exterior_faces(
            f,
            material,
            node_index,
            element_id,
            105,
        )
        surface_written[105] = element_id - before

        # Volumes.
        volume_first_id = element_id

        element_id, region_counts = write_volume_elements(
            f,
            material,
            node_index,
            volume_first_id,
            affine,
        )

        f.write("$EndElements\n")

    if element_id != total_count + 1:
        raise RuntimeError(
            f"Final element ID mismatch: "
            f"expected={total_count + 1}, actual={element_id}"
        )

    for tag, expected in interface_counts.items():
        actual = surface_written.get(tag, 0)
        if actual != expected:
            raise RuntimeError(
                f"Surface count mismatch for {SURFACE_NAMES[tag]}: "
                f"expected={expected:,}, actual={actual:,}"
            )

    if sum(region_counts.values()) != volume_count:
        raise RuntimeError("Region HEX8 counts do not sum to total volume count.")

    log(f"Mesh written successfully: {MESH_FILE}")

    return {
        "mesh_file": str(MESH_FILE),
        "format": "Gmsh MSH 2.2 ASCII",
        "volume_element_type": "HEX8",
        "gmsh_volume_element_type": GMSH_HEX8,
        "surface_element_type": "QUAD4",
        "gmsh_surface_element_type": GMSH_QUAD4,
        "nodes": node_count,
        "volume_elements": volume_count,
        "surface_elements": surface_count,
        "total_elements": total_count,
        "region_elements": {
            MATERIAL_NAMES[mid]: region_counts[mid]
            for mid in range(1, 6)
        },
        "surface_elements": {
            SURFACE_NAMES[tag]: interface_counts[tag]
            for tag in sorted(interface_counts)
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    start_time = time.perf_counter()

    section("MRI-ONLY GMSH FEM MESH GENERATION")
    log("5-REGION CONFORMING VOXEL FEM")
    log("HEX8 VOLUME + QUAD4 INTERFACE MESH")
    log("ORIGINAL MRI MASKS ARE READ-ONLY")
    log("ORIGINAL TUMOR SEGMENTATION IS READ-ONLY")
    log("NO MORPHOLOGICAL OPERATIONS")
    log("NO ADDITIONAL PADDING")
    log("USING PREPROCESSING-DERIVED FEM DOMAIN")

    section("PATHS")
    log(f"Script directory:  {SCRIPT_DIR}")
    log(f"Model root:        {MODEL_ROOT}")
    log(f"Dataset:            {DATASET_DIR}")
    log(f"Mask directory:     {MASK_DIR}")
    log(f"Material labels:    {MATERIAL_LABEL_FILE}")
    log(f"Raw tumor:          {RAW_TUMOR_SOURCE}")
    log(f"Output directory:   {RESULT_DIR}")
    log(f"Gmsh mesh:          {MESH_FILE}")

    prepare_directories()
    check_required_files()

    section("LOADING PREPROCESSING-DERIVED MATERIAL MASKS")

    mask_files = {
        "brain": BRAIN_MASK_FILE,
        "tumor": TUMOR_MASK_FILE,
        "csf": CSF_MASK_FILE,
        "skull": SKULL_MASK_FILE,
        "scalp": SCALP_MASK_FILE,
    }

    masks = {}
    affines = {}
    reference_affine = None
    reference_header = None

    for name, path in mask_files.items():
        mask, affine, header = load_binary_mask(path, name)
        masks[name] = mask
        affines[name] = affine

        if reference_affine is None:
            reference_affine = affine
            reference_header = header

    validate_mask_geometry(
        masks,
        affines,
        reference_affine,
    )

    spacing = voxel_spacing_from_affine(reference_affine)
    determinant = float(np.linalg.det(reference_affine[:3, :3]))

    section("VOXEL GEOMETRY")
    log(f"Shape:   {masks['brain'].shape}")
    log(
        f"Spacing: {spacing[0]:.6f} x "
        f"{spacing[1]:.6f} x {spacing[2]:.6f} mm"
    )
    log(f"Affine determinant: {determinant:.12g}")

    if abs(determinant) < 1e-12:
        raise RuntimeError("Invalid singular NIfTI affine.")

    validate_mask_exclusivity(masks)

    section("LOADING MATERIAL LABELS")

    material, material_affine, material_header = load_material_labels()

    validate_material_geometry(
        material,
        material_affine,
        masks["brain"].shape,
        reference_affine,
    )

    validate_material_against_masks(material, masks)

    section("MATERIAL STATISTICS")

    region_voxels = {}
    voxel_volume_mm3 = abs(determinant)

    for mid in range(1, 6):
        count = int(np.count_nonzero(material == mid))
        region_voxels[mid] = count

        volume_ml = count * voxel_volume_mm3 / 1000.0

        log(
            f"{MATERIAL_NAMES[mid]:<15}: "
            f"{count:,} voxels = {volume_ml:.3f} mL"
        )

    outside_voxels = int(np.count_nonzero(material == 0))
    log(f"Outside         : {outside_voxels:,} voxels")

    tumor_check = verify_tumor_exact(
        material,
        reference_affine,
    )

    interface_info = validate_material_interfaces(material)
    exterior_counts = count_exterior_faces(material)

    # Cross-check the Scalp exterior count independently.
    scalp_exterior = count_scalp_exterior_faces(material)
    if scalp_exterior != exterior_counts[5]:
        raise RuntimeError(
            "Independent Scalp exterior count mismatch."
        )

    section("BUILDING FEM NODE GRID")
    timer = Timer("Node-grid construction")
    used, node_index, node_count = build_node_index(material)
    timer.stop()

    interface_counts = {
        tag: int(info["count"])
        for tag, info in interface_info.items()
    }
    interface_counts[105] = int(exterior_counts[5])

    volume_count = int(np.count_nonzero(material > 0))
    surface_count = int(sum(interface_counts.values()))
    total_count = volume_count + surface_count

    section("MESH SIZE")
    log(f"FEM nodes:      {node_count:,}")
    log(f"HEX8 elements:  {volume_count:,}")
    log(f"QUAD4 elements: {surface_count:,}")
    log(f"Total elements: {total_count:,}")

    timer = Timer("Gmsh mesh writing")
    mesh_info = write_gmsh_mesh(
        material,
        used,
        node_index,
        node_count,
        reference_affine,
        interface_counts,
    )
    timer.stop()

    write_material_output(
        material,
        reference_affine,
        reference_header,
    )

    write_region_csv(region_voxels)
    write_interface_csv(interface_counts)

    summary = dict(mesh_info)
    summary.update(
        {
            "dataset_directory": str(DATASET_DIR),
            "mask_directory": str(MASK_DIR),
            "input_masks": {
                name: str(path)
                for name, path in mask_files.items()
            },
            "material_labels_input": str(MATERIAL_LABEL_FILE),
            "original_tumor_source": str(RAW_TUMOR_SOURCE),
            "shape": list(material.shape),
            "voxel_spacing_mm": [float(x) for x in spacing],
            "affine": reference_affine.tolist(),
            "affine_determinant_mm3": determinant,
            "voxel_volume_mm3": voxel_volume_mm3,
            "region_voxels": {
                MATERIAL_NAMES[mid]: region_voxels[mid]
                for mid in range(1, 6)
            },
            "outside_voxels": outside_voxels,
            "interface_validation": interface_info,
            "exterior_faces": {
                MATERIAL_NAMES[mid]: int(exterior_counts[mid])
                for mid in range(1, 6)
            },
            "tumor_preservation": tumor_check,
            "additional_padding_added_by_solver": False,
            "morphological_operations_applied_by_solver": False,
            "source_masks_modified": False,
            "conforming_voxel_mesh": True,
            "physical_volume_ids": {
                str(mid): MATERIAL_NAMES[mid]
                for mid in range(1, 6)
            },
            "physical_surface_ids": {
                str(tag): SURFACE_NAMES[tag]
                for tag in sorted(SURFACE_NAMES)
            },
        }
    )

    preprocess_report = load_preprocess_report()
    if preprocess_report is not None:
        summary["preprocess_report"] = preprocess_report

    with MESH_SUMMARY_FILE.open("w") as f:
        json.dump(summary, f, indent=2)

    # Stable machine-readable companion files required by downstream stages.
    with MESH_REPORT_FILE.open("w") as f:
        json.dump(summary, f, indent=2)
    with REGION_TAGS_FILE.open("w") as f:
        json.dump({
            "physical_volume_ids": {str(mid): MATERIAL_NAMES[mid] for mid in range(1, 6)},
            "physical_surface_ids": {str(tag): SURFACE_NAMES[tag] for tag in sorted(SURFACE_NAMES)},
        }, f, indent=2)

    section("FINAL VALIDATION")

    if np.any(node_index[used] < 1):
        raise RuntimeError("Invalid FEM node IDs detected.")

    if np.any(node_index[used] > node_count):
        raise RuntimeError("FEM node ID exceeds node count.")

    if sum(region_voxels.values()) != volume_count:
        raise RuntimeError("Region voxel counts do not equal HEX8 count.")

    if not tumor_check["exact_match"]:
        raise RuntimeError("Final tumor preservation failed.")

    for tag in range(101, 106):
        if interface_counts[tag] != summary["surface_elements"][SURFACE_NAMES[tag]]:
            raise RuntimeError(
                f"Final surface mismatch for {SURFACE_NAMES[tag]}."
            )

    for mid in range(1, 5):
        if exterior_counts[mid] != 0:
            raise RuntimeError(
                f"Final exterior validation failed for {MATERIAL_NAMES[mid]}."
            )

    if exterior_counts[5] <= 0:
        raise RuntimeError("Final exterior validation failed for Scalp.")

    log("Node validation:          PASS")
    log("Region-count validation:  PASS")
    log("Tumor validation:         PASS")
    log("Interface validation:     PASS")
    log("Exterior validation:      PASS")

    elapsed = time.perf_counter() - start_time

    section("FINAL FEM MESH STATISTICS")
    log(f"Domain shape: {material.shape}")
    log(
        f"Voxel spacing: "
        f"{spacing[0]:.6f} x {spacing[1]:.6f} x {spacing[2]:.6f} mm"
    )
    log("")
    log("Material regions:")

    for mid in range(1, 6):
        log(
            f"  {MATERIAL_NAMES[mid]:<15}: "
            f"{region_voxels[mid]:,} HEX8"
        )

    log("")
    log("Interfaces:")

    for tag in sorted(interface_counts):
        log(
            f"  {SURFACE_NAMES[tag]:<25}: "
            f"{interface_counts[tag]:,} QUAD4"
        )

    log("")
    log(f"Mesh file: {MESH_FILE}")
    log(f"Summary:   {MESH_SUMMARY_FILE}")
    log(f"Runtime:   {elapsed:.2f} seconds")

    log("")
    log("==============================================================")
    log("SUCCESS: MRI-only 5-region FEM mesh generation completed.")
    log("==============================================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fatal("Interrupted by user.")
    except Exception as exc:
        fatal(str(exc))
