#!/usr/bin/env python3
"""
preprocess.py

MRI-only preprocessing for a 5-region computational FEM head model.

Regions
-------
1 = BrainTissue
2 = Tumor
3 = CSF
4 = Skull
5 = Scalp

CORRECTED COMPUTATIONAL HIERARCHY
---------------------------------

                    OUTSIDE
                       |
                  +---------+
                  |  Scalp  |  4 vox
                  +---------+
                  |  Skull  |  3 vox
                  +---------+
                  |   CSF   |  2 vox
                  +---------+
                  |  Brain  |
                  |         |
                  | Tumor   |
                  +---------+

The hierarchy is constructed from the OUTER MRI HEAD MASK.

The brain envelope is NOT estimated using a low MRI intensity
percentile.

Instead:

    head_mask
        |
        +-- erode by:
              CSF thickness
            + Skull thickness
            + Scalp thickness
        |
        v
    brain_envelope

The computational shells are therefore explicitly inside the
outer MRI head boundary.

IMPORTANT
---------
- Original MRI files are READ-ONLY.
- Original tumor segmentation is READ-ONLY.
- Tumor voxels are preserved exactly.
- No morphology is applied to the original tumor mask.
- No morphology is applied directly to MASK.nii.gz.
- All computational masks are DERIVED masks.
- The derived computational domain has an explicit outside margin.
- BrainTissue, Tumor, CSF, Skull, and Scalp are mutually exclusive.
- Only Scalp may touch the computational outside.
- Tumor must be completely contained inside BrainTissue + Tumor.
- Unexpected direct material interfaces cause failure.

OUTPUT
------

preprocess/masks/
    brain_mask.nii.gz
    tumor_mask.nii.gz
    csf_mask.nii.gz
    skull_mask.nii.gz
    scalp_mask.nii.gz
    head_mask.nii.gz
    brain_envelope_mask.nii.gz
    material_labels.nii.gz
    material_labels_original.nii.gz

preprocess/preprocess_report.json

The original MRI and tumor files are never modified.
"""

from pathlib import Path
import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

PATIENT_DIR = (
    ROOT_DIR
    / "EGD-0001"
    / "MR_EGD-0001"
)

T1_PATH = (
    PATIENT_DIR
    / "1_T1"
    / "NIFTI"
    / "T1.nii.gz"
)

T1GD_PATH = (
    PATIENT_DIR
    / "2_T1GD"
    / "NIFTI"
    / "T1GD.nii.gz"
)

T2_PATH = (
    PATIENT_DIR
    / "3_T2"
    / "NIFTI"
    / "T2.nii.gz"
)

FLAIR_PATH = (
    PATIENT_DIR
    / "4_FLAIR"
    / "NIFTI"
    / "FLAIR.nii.gz"
)

MASK_PATH = (
    PATIENT_DIR
    / "5_MASK"
    / "NIFTI"
    / "MASK.nii.gz"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "preprocess"
    / "masks"
)

REPORT_PATH = (
    ROOT_DIR
    / "preprocess"
    / "preprocess_report.json"
)

HD_BET_MASK_PATH = OUTPUT_DIR / "brain_extraction_hdbet.nii.gz"


# ============================================================
# COMPUTATIONAL PARAMETERS
# ============================================================

CSF_THICKNESS_MM = 2.0
SKULL_THICKNESS_MM = 3.0
SCALP_THICKNESS_MM = 4.0

EXTRA_OUTER_MARGIN_VOXELS = 2


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(f"[preprocess] {message}")


def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


# ============================================================
# JSON SERIALIZATION
# ============================================================

def json_default(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        f"is not JSON serializable"
    )


# ============================================================
# INPUT CHECKING
# ============================================================

def check_input_files():

    paths = {
        "T1": T1_PATH,
        "T1GD": T1GD_PATH,
        "T2": T2_PATH,
        "FLAIR": FLAIR_PATH,
        "MASK": MASK_PATH,
    }

    missing = []

    for name, path in paths.items():

        if not path.exists():
            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "Missing input files:\n"
            + "\n".join(missing)
        )


# ============================================================
# NIFTI LOADING
# ============================================================

def load_nifti(path):

    img = nib.load(
        str(path)
    )

    data = np.asarray(
        img.get_fdata(
            dtype=np.float32
        )
    )

    spacing = tuple(
        float(x)
        for x in img.header.get_zooms()[:3]
    )

    return (
        data,
        img.affine.copy(),
        spacing,
        img.header.copy()
    )


# ============================================================
# GEOMETRY CHECK
# ============================================================

def check_same_geometry(
    reference_name,
    reference_data,
    reference_affine,
    data_name,
    data,
    affine
):

    if reference_data.shape != data.shape:

        raise ValueError(
            f"Geometry mismatch:\n"
            f"{reference_name}: "
            f"{reference_data.shape}\n"
            f"{data_name}: "
            f"{data.shape}"
        )

    if not np.allclose(
        reference_affine,
        affine,
        atol=1e-4
    ):

        raise ValueError(
            f"Affine mismatch between "
            f"{reference_name} and {data_name}"
        )


# ============================================================
# ROBUST NORMALIZATION
# ============================================================

def robust_normalize(volume):

    finite = volume[
        np.isfinite(volume)
    ]

    if finite.size == 0:

        return np.zeros(
            volume.shape,
            dtype=np.float32
        )

    p1 = float(
        np.percentile(
            finite,
            1.0
        )
    )

    p99 = float(
        np.percentile(
            finite,
            99.0
        )
    )

    if p99 <= p1:

        return np.zeros(
            volume.shape,
            dtype=np.float32
        )

    normalized = (
        volume - p1
    ) / (
        p99 - p1
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# MORPHOLOGY HELPERS
# ============================================================

def make_structure():

    # 6-connected 3-D neighborhood.
    #
    # This is appropriate for the voxel FEM construction
    # because dilation/erosion proceeds face-to-face rather
    # than diagonally through corners.

    return ndimage.generate_binary_structure(
        rank=3,
        connectivity=1
    )


def dilate_mask(
    mask,
    iterations
):

    iterations = int(
        round(iterations)
    )

    if iterations <= 0:

        return np.asarray(
            mask,
            dtype=bool
        )

    return ndimage.binary_dilation(
        np.asarray(mask, dtype=bool),
        structure=make_structure(),
        iterations=iterations
    )


def erode_mask(
    mask,
    iterations
):

    iterations = int(
        round(iterations)
    )

    if iterations <= 0:

        return np.asarray(
            mask,
            dtype=bool
        )

    return ndimage.binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=make_structure(),
        iterations=iterations
    )


def close_mask(
    mask,
    iterations=1
):

    if iterations <= 0:

        return np.asarray(
            mask,
            dtype=bool
        )

    return ndimage.binary_closing(
        np.asarray(mask, dtype=bool),
        structure=make_structure(),
        iterations=iterations
    )


def remove_small_components(
    mask,
    min_size=100
):

    mask = np.asarray(
        mask,
        dtype=bool
    )

    labels, num = ndimage.label(
        mask,
        structure=make_structure()
    )

    if num == 0:

        return np.zeros_like(
            mask,
            dtype=bool
        )

    counts = np.bincount(
        labels.ravel()
    )

    keep = counts >= int(
        min_size
    )

    keep[0] = False

    return keep[labels]


def keep_largest_component(mask):

    mask = np.asarray(
        mask,
        dtype=bool
    )

    labels, num = ndimage.label(
        mask,
        structure=make_structure()
    )

    if num == 0:

        return np.zeros_like(
            mask,
            dtype=bool
        )

    counts = np.bincount(
        labels.ravel()
    )

    counts[0] = 0

    largest = int(
        np.argmax(counts)
    )

    return (
        labels == largest
    )


# ============================================================
# HEAD ESTIMATION
# ============================================================

def estimate_head_mask(
    t1,
    t1gd,
    t2,
    flair
):
    """
    Estimate the outer MRI head envelope.

    This is deliberately used as the OUTER computational
    boundary.

    It is NOT interpreted as an anatomical brain mask.
    """

    section(
        "ESTIMATING OUTER MRI HEAD"
    )

    log(
        "Building combined MRI volume."
    )

    combined = (
        t1
        + t1gd
        + t2
        + flair
    ) / 4.0

    combined = np.nan_to_num(
        combined,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    positive = combined[
        combined > 0
    ]

    if positive.size == 0:

        raise RuntimeError(
            "Combined MRI contains no positive voxels."
        )

    log(
        f"Positive MRI voxels: "
        f"{positive.size:,}"
    )

    log(
        "Calculating Otsu threshold."
    )

    otsu_threshold = float(
        threshold_otsu(
            positive
        )
    )

    # Conservative head threshold.
    #
    # The goal is to capture the complete outer head
    # envelope rather than to classify brain tissue.

    threshold = (
        otsu_threshold * 0.50
    )

    log(
        f"Otsu threshold: "
        f"{otsu_threshold:.8f}"
    )

    log(
        f"Head threshold: "
        f"{threshold:.8f}"
    )

    head = (
        combined > threshold
    )

    log(
        f"Raw head voxels: "
        f"{np.count_nonzero(head):,}"
    )

    log(
        "Closing small discontinuities."
    )

    head = close_mask(
        head,
        iterations=2
    )

    log(
        "Filling internal holes."
    )

    head = ndimage.binary_fill_holes(
        head
    )

    log(
        "Removing small components."
    )

    head = remove_small_components(
        head,
        min_size=1000
    )

    log(
        "Keeping largest connected head component."
    )

    head = keep_largest_component(
        head
    )

    head = head.astype(bool)

    log(
        f"Final head voxels: "
        f"{np.count_nonzero(head):,}"
    )

    if not np.any(head):

        raise RuntimeError(
            "Outer MRI head mask is empty."
        )

    return (
        head,
        combined.astype(np.float32),
        threshold,
        otsu_threshold
    )


def estimate_anatomical_brain_mask(combined, head_mask, tumor_mask):
    """Derive an MRI-only inner brain envelope from the four scans."""
    section("ESTIMATING MRI-ONLY ANATOMICAL BRAIN ENVELOPE")
    if HD_BET_MASK_PATH.exists():
        nii = nib.load(str(HD_BET_MASK_PATH))
        brain = np.asarray(nii.dataobj) > 0
        if brain.shape != head_mask.shape:
            raise RuntimeError("HD-BET mask shape does not match MRI geometry")
        log(f"Using validated HD-BET mask: {HD_BET_MASK_PATH}")
        # Preserve exact tumor voxels and add a derived two-voxel cushion so
        # the tumor cannot touch the generated CSF shell.
        brain = (brain | dilate_mask(tumor_mask, 2)) & head_mask
        return brain.astype(bool), {"method": "HD-BET CPU", "mask_file": str(HD_BET_MASK_PATH)}

    inner = erode_mask(head_mask, 4)
    values = combined[inner & np.isfinite(combined) & (combined > 0)]
    if values.size == 0:
        raise RuntimeError("No positive MRI voxels remain inside the inner head")
    otsu = float(threshold_otsu(values))
    threshold = 0.92 * otsu
    log(f"Inner-head Otsu threshold: {otsu:.8f}")
    log(f"Brain threshold: {threshold:.8f}")
    brain = (combined >= threshold) & inner
    brain = close_mask(brain, 2)
    brain = ndimage.binary_fill_holes(brain)
    brain = keep_largest_component(brain)
    brain |= tumor_mask
    if not np.any(brain):
        raise RuntimeError("MRI-only anatomical brain envelope is empty")
    log(f"Derived anatomical brain-envelope voxels: {np.count_nonzero(brain):,}")
    return brain.astype(bool), {"method": "inner-head Otsu", "inner_erosion_voxels": 4, "otsu_threshold": otsu, "threshold": threshold}


# ============================================================
# MM -> VOXEL CONVERSION
# ============================================================

def thickness_mm_to_voxels(
    thickness_mm,
    spacing
):

    min_spacing = min(
        float(x)
        for x in spacing
    )

    if min_spacing <= 0:

        raise ValueError(
            f"Invalid voxel spacing: {spacing}"
        )

    return max(
        1,
        int(
            np.ceil(
                float(thickness_mm)
                / min_spacing
            )
        )
    )


# ============================================================
# DERIVED CANVAS MARGIN
# ============================================================

def calculate_domain_margin(
    spacing
):

    csf_voxels = thickness_mm_to_voxels(
        CSF_THICKNESS_MM,
        spacing
    )

    skull_voxels = thickness_mm_to_voxels(
        SKULL_THICKNESS_MM,
        spacing
    )

    scalp_voxels = thickness_mm_to_voxels(
        SCALP_THICKNESS_MM,
        spacing
    )

    margin = (
        csf_voxels
        + skull_voxels
        + scalp_voxels
        + EXTRA_OUTER_MARGIN_VOXELS
    )

    return (
        csf_voxels,
        skull_voxels,
        scalp_voxels,
        margin
    )


# ============================================================
# PAD DERIVED COMPUTATIONAL INPUT
# ============================================================

def pad_derived_inputs(
    head_mask,
    tumor_mask,
    brain_mask,
    affine,
    margin
):
    """
    Create a larger DERIVED computational domain.

    Original arrays are never modified.

    The original voxel at [0,0,0] moves to
    [margin,margin,margin].

    The affine is translated so that physical coordinates
    of the original MRI voxels remain unchanged.
    """

    margin = int(margin)

    old_shape = head_mask.shape

    new_shape = tuple(
        int(n + 2 * margin)
        for n in old_shape
    )

    log(
        f"Original shape: {old_shape}"
    )

    log(
        f"Derived shape : {new_shape}"
    )

    log(
        f"Derived margin: {margin} voxels/side"
    )

    head_padded = np.zeros(
        new_shape,
        dtype=bool
    )

    tumor_padded = np.zeros(
        new_shape,
        dtype=bool
    )

    brain_padded = np.zeros(
        new_shape,
        dtype=bool
    )

    slices = tuple(
        slice(
            margin,
            margin + n
        )
        for n in old_shape
    )

    head_padded[slices] = (
        head_mask
    )

    tumor_padded[slices] = (
        tumor_mask
    )

    brain_padded[slices] = (
        brain_mask
    )

    new_affine = np.asarray(
        affine,
        dtype=float
    ).copy()

    offset = np.asarray(
        [
            margin,
            margin,
            margin
        ],
        dtype=float
    )

    new_affine[:3, 3] = (
        affine[:3, 3]
        - affine[:3, :3] @ offset
    )

    return (
        head_padded,
        tumor_padded,
        brain_padded,
        new_affine
    )


# ============================================================
# BUILD CORRECTED NESTED COMPUTATIONAL REGIONS
# ============================================================

def build_nested_regions(
    head_mask,
    tumor_mask,
    csf_voxels,
    skull_voxels,
    scalp_voxels,
    brain_source=None
):
    """
    Build the five mutually-exclusive computational regions.

    IMPORTANT:

    The brain envelope is derived by ERoding the OUTER HEAD
    by the total thickness required for:

        CSF + Skull + Scalp

    This is the key correction.

    Therefore:

        brain_envelope
            |
            +-- CSF
            |
            +-- Skull
            |
            +-- Scalp
            |
            +-- outside

    The tumor is not modified.

    The tumor must already be inside the brain envelope.
    """

    head_mask = np.asarray(
        head_mask,
        dtype=bool
    )

    if brain_source is None:
        brain_source = head_mask
    else:
        brain_source = np.asarray(brain_source, dtype=bool)

    tumor_mask = np.asarray(
        tumor_mask,
        dtype=bool
    )

    # --------------------------------------------------------
    # Total shell thickness
    # --------------------------------------------------------

    total_shell_voxels = (
        int(csf_voxels)
        + int(skull_voxels)
        + int(scalp_voxels)
    )

    log(
        f"Total shell thickness: "
        f"{total_shell_voxels} voxels"
    )

    # --------------------------------------------------------
    # Brain envelope
    # --------------------------------------------------------
    #
    # THIS IS THE CRITICAL CORRECTION.
    #
    # Do NOT estimate the brain using a low MRI intensity
    # threshold.
    #
    # Instead, reserve enough space inside the head for the
    # three computational shells.
    # --------------------------------------------------------

    # Search from the nominal shell reservation down to zero.  This preserves
    # the largest safe geometric erosion while never changing the tumor.
    erosion_history = []
    brain_envelope = None
    selected_erosion = None
    for erosion in range(total_shell_voxels, -1, -1):
        candidate = erode_mask(brain_source, erosion).astype(bool)
        outside = int(np.count_nonzero(tumor_mask & ~candidate))
        # A contained tumor can still touch the CSF shell if it lies on the
        # envelope boundary. Require one 6-connected voxel of brain tissue
        # between tumor and the outside of the candidate envelope.
        tumor_boundary = int(np.count_nonzero(
            dilate_mask(tumor_mask, 1) & ~candidate
        ))
        erosion_history.append({
            "erosion_voxels": int(erosion),
            "brain_envelope_voxels": int(np.count_nonzero(candidate)),
            "tumor_voxels_outside": outside,
            "tumor_boundary_contacts": tumor_boundary,
        })
        log(f"erosion = {erosion}: brain-envelope voxels = "
            f"{np.count_nonzero(candidate):,}; tumor outside = {outside:,}; "
            f"tumor boundary contacts = {tumor_boundary:,}")
        if (selected_erosion is None and np.any(candidate)
                and outside == 0 and tumor_boundary == 0):
            selected_erosion = erosion
            brain_envelope = candidate

    if brain_envelope is None:
        raise RuntimeError(
            "Tumor is not contained in the MRI head even at zero erosion. "
            "The tumor mask was not modified."
        )

    log(f"Selected safe brain-envelope erosion: {selected_erosion} voxels")

    # --------------------------------------------------------
    # CSF
    # --------------------------------------------------------
    #
    # Start from brain envelope and dilate outward.
    # --------------------------------------------------------

    csf_outer = dilate_mask(
        brain_envelope,
        csf_voxels
    ) & head_mask

    csf_mask = (
        csf_outer &
        ~brain_envelope
    )

    # --------------------------------------------------------
    # Skull
    # --------------------------------------------------------

    skull_outer = dilate_mask(
        csf_outer,
        skull_voxels
    )

    skull_mask = (
        skull_outer &
        ~csf_outer
    )

    # --------------------------------------------------------
    # Scalp
    # --------------------------------------------------------

    # The scalp must terminate at the MRI-derived outer head, not at a
    # nine-voxel dilation of the brain. This preserves the anatomical outer
    # contour while allowing CSF and skull to remain shells around the brain.
    scalp_outer = dilate_mask(head_mask, scalp_voxels)
    scalp_mask = scalp_outer & ~skull_outer

    # --------------------------------------------------------
    # Brain tissue
    # --------------------------------------------------------
    #
    # Tumor remains its own material.
    # --------------------------------------------------------

    brain_tissue = (
        brain_envelope &
        ~tumor_mask
    )

    # --------------------------------------------------------
    # Explicit exclusivity
    # --------------------------------------------------------

    brain_tissue &= ~tumor_mask

    csf_mask &= ~brain_tissue
    csf_mask &= ~tumor_mask

    skull_mask &= ~brain_tissue
    skull_mask &= ~tumor_mask
    skull_mask &= ~csf_mask

    scalp_mask &= ~brain_tissue
    scalp_mask &= ~tumor_mask
    scalp_mask &= ~csf_mask
    scalp_mask &= ~skull_mask

    return {
        "BrainTissue":
            brain_tissue.astype(bool),

        "Tumor":
            tumor_mask.astype(bool),

        "CSF":
            csf_mask.astype(bool),

        "Skull":
            skull_mask.astype(bool),

        "Scalp":
            scalp_mask.astype(bool),

        "brain_envelope":
            brain_envelope.astype(bool),

        "csf_outer":
            csf_outer.astype(bool),

        "skull_outer":
            skull_outer.astype(bool),

        "scalp_outer":
            scalp_outer.astype(bool),

        "selected_safe_erosion_voxels":
            int(selected_erosion),

        "erosion_search_history":
            erosion_history,
    }


# ============================================================
# MATERIAL LABEL VOLUME
# ============================================================

def build_material_labels(
    regions
):

    shape = regions[
        "BrainTissue"
    ].shape

    material = np.zeros(
        shape,
        dtype=np.uint8
    )

    material[
        regions["BrainTissue"]
    ] = 1

    material[
        regions["Tumor"]
    ] = 2

    material[
        regions["CSF"]
    ] = 3

    material[
        regions["Skull"]
    ] = 4

    material[
        regions["Scalp"]
    ] = 5

    return material


# ============================================================
# REGION BOUNDING BOX
# ============================================================

def mask_bbox(
    mask,
    spacing,
    affine
):

    coordinates = np.argwhere(
        mask
    )

    if coordinates.size == 0:

        return {
            "voxel_min": None,
            "voxel_max": None,
            "world_min_mm": None,
            "world_max_mm": None,
        }

    voxel_min = coordinates.min(
        axis=0
    )

    voxel_max = coordinates.max(
        axis=0
    )

    # Convert the 8 corners of the voxel-index bbox to
    # physical coordinates.

    mins = voxel_min.astype(float)
    maxs = voxel_max.astype(float)

    corners = []

    for x in [mins[0], maxs[0]]:
        for y in [mins[1], maxs[1]]:
            for z in [mins[2], maxs[2]]:

                point = np.asarray(
                    [x, y, z, 1.0],
                    dtype=float
                )

                world = (
                    affine @ point
                )[:3]

                corners.append(
                    world
                )

    corners = np.asarray(
        corners
    )

    world_min = corners.min(
        axis=0
    )

    world_max = corners.max(
        axis=0
    )

    return {
        "voxel_min": [
            int(x)
            for x in voxel_min
        ],

        "voxel_max": [
            int(x)
            for x in voxel_max
        ],

        "world_min_mm": [
            float(x)
            for x in world_min
        ],

        "world_max_mm": [
            float(x)
            for x in world_max
        ],
    }


# ============================================================
# REGION CENTROID
# ============================================================

def mask_centroid_world(
    mask,
    affine
):

    coordinates = np.argwhere(
        mask
    )

    if coordinates.size == 0:

        return None

    centroid_voxel = (
        coordinates.mean(
            axis=0
        )
    )

    point = np.asarray(
        [
            centroid_voxel[0],
            centroid_voxel[1],
            centroid_voxel[2],
            1.0,
        ],
        dtype=float
    )

    centroid_world = (
        affine @ point
    )[:3]

    return [
        float(x)
        for x in centroid_world
    ]


# ============================================================
# REGION DIAGNOSTICS
# ============================================================

def region_diagnostics(
    name,
    mask,
    spacing,
    affine
):

    count = int(
        np.count_nonzero(mask)
    )

    voxel_volume = (
        float(spacing[0])
        * float(spacing[1])
        * float(spacing[2])
    )

    bbox = mask_bbox(
        mask,
        spacing,
        affine
    )

    centroid = mask_centroid_world(
        mask,
        affine
    )

    return {
        "voxels": count,

        "volume_mm3":
            float(count * voxel_volume),

        "volume_ml":
            float(
                count
                * voxel_volume
                / 1000.0
            ),

        "bbox":
            bbox,

        "centroid_world_mm":
            centroid,
    }


# ============================================================
# PRINT REGION DIAGNOSTICS
# ============================================================

def print_region_diagnostics(
    regions,
    spacing,
    affine
):

    section(
        "STRONG REGION GEOMETRY DIAGNOSTICS"
    )

    names = [
        "BrainTissue",
        "Tumor",
        "CSF",
        "Skull",
        "Scalp",
    ]

    for name in names:

        stats = region_diagnostics(
            name,
            regions[name],
            spacing,
            affine
        )

        bbox = stats["bbox"]

        log(
            f"{name}"
        )

        log(
            f"  Voxels     : "
            f"{stats['voxels']:,}"
        )

        log(
            f"  Volume     : "
            f"{stats['volume_ml']:.3f} mL"
        )

        log(
            f"  Voxel min  : "
            f"{bbox['voxel_min']}"
        )

        log(
            f"  Voxel max  : "
            f"{bbox['voxel_max']}"
        )

        log(
            f"  World min  : "
            f"{bbox['world_min_mm']}"
        )

        log(
            f"  World max  : "
            f"{bbox['world_max_mm']}"
        )

        log(
            f"  Centroid   : "
            f"{stats['centroid_world_mm']}"
        )

        print()


# ============================================================
# MATERIAL EXCLUSIVITY CHECK
# ============================================================

def check_material_exclusivity(
    regions
):

    section(
        "CHECKING MATERIAL-MASK MUTUAL EXCLUSIVITY"
    )

    names = [
        "BrainTissue",
        "Tumor",
        "CSF",
        "Skull",
        "Scalp",
    ]

    total = np.zeros(
        regions[names[0]].shape,
        dtype=np.uint8
    )

    for name in names:

        total += (
            regions[name]
            .astype(np.uint8)
        )

    overlap_count = int(
        np.count_nonzero(
            total > 1
        )
    )

    occupied = int(
        np.count_nonzero(
            total > 0
        )
    )

    if overlap_count != 0:

        raise RuntimeError(
            f"Material masks overlap in "
            f"{overlap_count:,} voxels."
        )

    log(
        "Material masks are mutually exclusive."
    )

    log(
        f"Occupied material voxels: "
        f"{occupied:,}"
    )


# ============================================================
# TUMOR PRESERVATION
# ============================================================

def check_tumor_preservation(
    original_tumor,
    derived_tumor
):

    section(
        "TUMOR PRESERVATION CHECK"
    )

    original_count = int(
        np.count_nonzero(
            original_tumor
        )
    )

    derived_count = int(
        np.count_nonzero(
            derived_tumor
        )
    )

    count_difference = (
        derived_count
        - original_count
    )

    log(
        f"Original tumor voxels: "
        f"{original_count:,}"
    )

    log(
        f"Derived tumor voxels : "
        f"{derived_count:,}"
    )

    log(
        f"Voxel difference      : "
        f"{count_difference:,}"
    )

    if count_difference != 0:

        raise RuntimeError(
            "Tumor voxel count changed."
        )

    log(
        "Tumor voxel count preserved exactly."
    )


# ============================================================
# EMBEDDED TUMOR EXACT-MATCH
# ============================================================

def check_embedded_tumor_exact_match(
    original_tumor,
    derived_tumor,
    margin
):

    old_shape = original_tumor.shape

    slices = tuple(
        slice(
            margin,
            margin + n
        )
        for n in old_shape
    )

    embedded = (
        derived_tumor[slices]
    )

    exact = bool(
        np.array_equal(
            original_tumor,
            embedded
        )
    )

    differing = int(
        np.count_nonzero(
            original_tumor != embedded
        )
    )

    log(
        f"Embedded tumor exact match: "
        f"{exact}"
    )

    log(
        f"Differing tumor voxels: "
        f"{differing:,}"
    )

    if not exact:

        raise RuntimeError(
            "Derived FEM domain changed "
            "the original tumor segmentation."
        )


# ============================================================
# TUMOR CONTAINMENT
# ============================================================

def check_tumor_containment(
    tumor,
    brain_envelope
):

    section(
        "CHECKING TUMOR CONTAINMENT"
    )

    outside = (
        tumor &
        ~brain_envelope
    )

    count = int(
        np.count_nonzero(
            outside
        )
    )

    tumor_count = int(
        np.count_nonzero(
            tumor
        )
    )

    log(
        f"Tumor voxels       : "
        f"{tumor_count:,}"
    )

    log(
        f"Outside brain      : "
        f"{count:,}"
    )

    if count != 0:

        raise RuntimeError(
            "Tumor is not completely contained "
            "inside the brain envelope."
        )

    log(
        "Tumor containment passed."
    )


# ============================================================
# NESTING CHECK
# ============================================================

def check_nested_geometry(
    regions
):

    section(
        "CHECKING NESTED GEOMETRY"
    )

    brain = regions[
        "brain_envelope"
    ]

    csf_outer = regions[
        "csf_outer"
    ]

    skull_outer = regions[
        "skull_outer"
    ]

    scalp_outer = regions[
        "scalp_outer"
    ]

    checks = {

        "Brain inside CSF outer":
            np.count_nonzero(
                brain &
                ~csf_outer
            ),

        "CSF outer inside Skull outer":
            np.count_nonzero(
                csf_outer &
                ~skull_outer
            ),

        "Skull outer inside Scalp outer":
            np.count_nonzero(
                skull_outer &
                ~scalp_outer
            ),
    }

    for name, count in checks.items():

        count = int(count)

        log(
            f"{name:<40}: "
            f"{count:,} violations"
        )

        if count != 0:

            raise RuntimeError(
                f"Nested geometry failed: {name}"
            )

    log(
        "Nested geometry validation passed."
    )


# ============================================================
# EXPECTED MATERIAL INTERFACES
# ============================================================

EXPECTED_INTERFACES = {
    frozenset((1, 2)),
    frozenset((1, 3)),
    frozenset((3, 4)),
    frozenset((4, 5)),
}


# ============================================================
# MATERIAL INTERFACE COLLECTION
# ============================================================

def collect_material_interfaces(
    material
):

    interfaces = set()

    def collect(a, b):

        valid = (
            (a > 0)
            &
            (b > 0)
            &
            (a != b)
        )

        labels_a = a[valid]
        labels_b = b[valid]

        for x, y in zip(
            labels_a.tolist(),
            labels_b.tolist()
        ):

            interfaces.add(
                frozenset(
                    (
                        int(x),
                        int(y)
                    )
                )
            )

    collect(
        material[:-1, :, :],
        material[1:, :, :]
    )

    collect(
        material[:, :-1, :],
        material[:, 1:, :]
    )

    collect(
        material[:, :, :-1],
        material[:, :, 1:]
    )

    return interfaces


# ============================================================
# MATERIAL INTERFACE COUNTS
# ============================================================

def count_material_interfaces(
    material
):

    counts = {}

    def add_counts(a, b):

        valid = (
            (a > 0)
            &
            (b > 0)
            &
            (a != b)
        )

        if not np.any(valid):
            return

        pairs_a = a[valid].astype(
            np.int16
        )

        pairs_b = b[valid].astype(
            np.int16
        )

        for x, y in zip(
            pairs_a.tolist(),
            pairs_b.tolist()
        ):

            pair = tuple(
                sorted(
                    (
                        int(x),
                        int(y)
                    )
                )
            )

            counts[pair] = (
                counts.get(pair, 0)
                + 1
            )

    add_counts(
        material[:-1, :, :],
        material[1:, :, :]
    )

    add_counts(
        material[:, :-1, :],
        material[:, 1:, :]
    )

    add_counts(
        material[:, :, :-1],
        material[:, :, 1:]
    )

    return counts


# ============================================================
# VALIDATE MATERIAL INTERFACES
# ============================================================

def validate_nested_material_structure(
    material
):

    section(
        "VALIDATING MATERIAL INTERFACES"
    )

    interfaces = (
        collect_material_interfaces(
            material
        )
    )

    log(
        "Detected material interfaces:"
    )

    names = {
        1: "BrainTissue",
        2: "Tumor",
        3: "CSF",
        4: "Skull",
        5: "Scalp",
    }

    for pair in sorted(
        interfaces,
        key=lambda x: tuple(x)
    ):

        values = sorted(
            list(pair)
        )

        log(
            f"  {names[values[0]]}({values[0]}) "
            f"<-> "
            f"{names[values[1]]}({values[1]})"
        )

    unexpected = (
        interfaces
        - EXPECTED_INTERFACES
    )

    missing = (
        EXPECTED_INTERFACES
        - interfaces
    )

    if unexpected:

        lines = []

        for pair in sorted(
            unexpected,
            key=lambda x: tuple(x)
        ):

            values = sorted(
                list(pair)
            )

            lines.append(
                f"{names[values[0]]}({values[0]}) "
                f"<-> "
                f"{names[values[1]]}({values[1]})"
            )

        raise RuntimeError(
            "Unexpected material interfaces:\n"
            + "\n".join(lines)
        )

    if missing:

        lines = []

        for pair in sorted(
            missing,
            key=lambda x: tuple(x)
        ):

            values = sorted(
                list(pair)
            )

            lines.append(
                f"{names[values[0]]}({values[0]}) "
                f"<-> "
                f"{names[values[1]]}({values[1]})"
            )

        raise RuntimeError(
            "Expected material interfaces are missing:\n"
            + "\n".join(lines)
        )

    log(
        "All expected material interfaces are present."
    )

    log(
        "No forbidden direct interfaces detected."
    )


# ============================================================
# EXTERIOR CONTACT COUNTS
# ============================================================

def exterior_contact_counts(
    material
):

    """
    Count material-to-outside faces.

    Outside = label 0.

    Only Scalp (5) is allowed to touch outside.
    """

    padded = np.pad(
        material,
        1,
        mode="constant",
        constant_values=0
    )

    center = padded[
        1:-1,
        1:-1,
        1:-1
    ]

    counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    neighbors = [

        padded[
            :-2,
            1:-1,
            1:-1
        ],

        padded[
            2:,
            1:-1,
            1:-1
        ],

        padded[
            1:-1,
            :-2,
            1:-1
        ],

        padded[
            1:-1,
            2:,
            1:-1
        ],

        padded[
            1:-1,
            1:-1,
            :-2
        ],

        padded[
            1:-1,
            1:-1,
            2:
        ],
    ]

    for neighbor in neighbors:

        outside = (
            neighbor == 0
        )

        for label in range(1, 6):

            counts[label] += int(
                np.count_nonzero(
                    (center == label)
                    &
                    outside
                )
            )

    return counts


# ============================================================
# VALIDATE EXTERIOR
# ============================================================

def validate_exterior(
    material
):

    section(
        "VALIDATING COMPUTATIONAL EXTERIOR"
    )

    counts = (
        exterior_contact_counts(
            material
        )
    )

    names = {
        1: "BrainTissue",
        2: "Tumor",
        3: "CSF",
        4: "Skull",
        5: "Scalp",
    }

    for label in range(1, 6):

        log(
            f"{names[label]:<15}: "
            f"{counts[label]:,} "
            f"outside faces"
        )

    non_scalp = {
        label: count
        for label, count
        in counts.items()
        if label != 5
        and count > 0
    }

    if non_scalp:

        lines = []

        for label, count in (
            non_scalp.items()
        ):

            lines.append(
                f"{names[label]}: "
                f"{count:,} faces"
            )

        raise RuntimeError(
            "Non-scalp material touches "
            "the computational outside:\n"
            + "\n".join(lines)
        )

    if counts[5] == 0:

        raise RuntimeError(
            "Scalp does not touch the computational "
            "outside. The derived domain is invalid."
        )

    log(
        "Exterior validation passed."
    )

    log(
        "Scalp is the only material touching outside."
    )

    return counts


# ============================================================
# MATERIAL STATISTICS
# ============================================================

def voxel_volume_mm3(
    voxel_count,
    spacing
):

    volume = (
        float(spacing[0])
        * float(spacing[1])
        * float(spacing[2])
    )

    return (
        int(voxel_count)
        * volume
    )


def voxels_to_ml(
    voxel_count,
    spacing
):

    return (
        voxel_volume_mm3(
            voxel_count,
            spacing
        )
        / 1000.0
    )


def mask_statistics(
    mask,
    spacing
):

    count = int(
        np.count_nonzero(mask)
    )

    return {
        "voxels":
            count,

        "volume_mm3":
            float(
                voxel_volume_mm3(
                    count,
                    spacing
                )
            ),

        "volume_ml":
            float(
                voxels_to_ml(
                    count,
                    spacing
                )
            ),
    }


# ============================================================
# SAVE NIFTI
# ============================================================

def save_nifti(
    data,
    affine,
    reference_header,
    path,
    dtype
):

    array = np.asarray(
        data,
        dtype=dtype
    )

    header = reference_header.copy()

    header.set_data_dtype(
        dtype
    )

    image = nib.Nifti1Image(
        array,
        affine,
        header=header
    )

    nib.save(
        image,
        str(path)
    )


# ============================================================
# MATERIAL STATISTICS
# ============================================================

def material_statistics(
    material,
    spacing
):

    names = {
        0: "Outside",
        1: "BrainTissue",
        2: "Tumor",
        3: "CSF",
        4: "Skull",
        5: "Scalp",
    }

    result = {}

    total = material.size

    for label, name in names.items():

        count = int(
            np.count_nonzero(
                material == label
            )
        )

        result[name] = {

            "label":
                label,

            "voxels":
                count,

            "volume_mm3":
                float(
                    voxel_volume_mm3(
                        count,
                        spacing
                    )
                ),

            "volume_ml":
                float(
                    voxels_to_ml(
                        count,
                        spacing
                    )
                ),
        }

    result["_total_voxels"] = int(
        total
    )

    return result


# ============================================================
# BUILD DETAILED REGION REPORT
# ============================================================

def build_region_report(
    regions,
    spacing,
    affine
):

    names = [
        "BrainTissue",
        "Tumor",
        "CSF",
        "Skull",
        "Scalp",
    ]

    result = {}

    for name in names:

        result[name] = (
            region_diagnostics(
                name,
                regions[name],
                spacing,
                affine
            )
        )

    return result


# ============================================================
# REPORT
# ============================================================

def build_report(
    original_shape,
    final_shape,
    spacing,
    margin,
    csf_voxels,
    skull_voxels,
    scalp_voxels,
    head_threshold,
    otsu_threshold,
    original_tumor,
    derived_tumor,
    material,
    original_material,
    exterior_counts,
    original_affine,
    derived_affine,
    regions
):

    tumor_original_count = int(
        np.count_nonzero(
            original_tumor
        )
    )

    tumor_final_count = int(
        np.count_nonzero(
            derived_tumor
        )
    )

    interface_counts = (
        count_material_interfaces(
            material
        )
    )

    interface_report = {}

    interface_names = {
        (1, 2):
            "BrainTissue <-> Tumor",

        (1, 3):
            "BrainTissue <-> CSF",

        (3, 4):
            "CSF <-> Skull",

        (4, 5):
            "Skull <-> Scalp",
    }

    for pair, name in (
        interface_names.items()
    ):

        interface_report[name] = int(
            interface_counts.get(
                pair,
                0
            )
        )

    report = {

        "dataset": {

            "patient":
                "EGD-0001",

            "modality":
                "MRI",

            "modalities": [
                "T1",
                "T1GD",
                "T2",
                "FLAIR",
            ],
        },

        "input": {

            "T1":
                str(T1_PATH),

            "T1GD":
                str(T1GD_PATH),

            "T2":
                str(T2_PATH),

            "FLAIR":
                str(FLAIR_PATH),

            "MASK":
                str(MASK_PATH),
        },

        "geometry": {

            "original_shape": [
                int(x)
                for x in original_shape
            ],

            "derived_shape": [
                int(x)
                for x in final_shape
            ],

            "spacing_mm": [
                float(x)
                for x in spacing
            ],

            "voxel_volume_mm3":
                float(
                    spacing[0]
                    * spacing[1]
                    * spacing[2]
                ),

            "original_affine":
                original_affine.tolist(),

            "derived_affine":
                derived_affine.tolist(),
        },

        "computational_parameters": {

            "csf_thickness_mm":
                float(
                    CSF_THICKNESS_MM
                ),

            "skull_thickness_mm":
                float(
                    SKULL_THICKNESS_MM
                ),

            "scalp_thickness_mm":
                float(
                    SCALP_THICKNESS_MM
                ),

            "csf_dilation_voxels":
                int(csf_voxels),

            "skull_dilation_voxels":
                int(skull_voxels),

            "scalp_dilation_voxels":
                int(scalp_voxels),

            "total_shell_voxels":
                int(
                    csf_voxels
                    + skull_voxels
                    + scalp_voxels
                ),

            "extra_outer_margin_voxels":
                int(
                    EXTRA_OUTER_MARGIN_VOXELS
                ),

            "total_derived_margin_voxels":
                int(margin),

            "otsu_threshold":
                float(otsu_threshold),

            "head_threshold":
                float(head_threshold),
        },

        "tumor_preservation": {

            "original_voxels":
                tumor_original_count,

            "derived_voxels":
                tumor_final_count,

            "voxel_count_difference":
                int(
                    tumor_final_count
                    - tumor_original_count
                ),

            "exact_count_match":
                bool(
                    tumor_original_count
                    == tumor_final_count
                ),
        },

        "region_geometry":
            build_region_report(
                regions,
                spacing,
                derived_affine
            ),

        "original_material_statistics":
            material_statistics(
                original_material,
                spacing
            ),

        "derived_material_statistics":
            material_statistics(
                material,
                spacing
            ),

        "material_interfaces":
            interface_report,

        "derived_exterior_faces": {

            "BrainTissue":
                int(exterior_counts[1]),

            "Tumor":
                int(exterior_counts[2]),

            "CSF":
                int(exterior_counts[3]),

            "Skull":
                int(exterior_counts[4]),

            "Scalp":
                int(exterior_counts[5]),
        },

        "model_definition": {

            "regions": {

                "1":
                    "BrainTissue",

                "2":
                    "Tumor",

                "3":
                    "CSF",

                "4":
                    "Skull",

                "5":
                    "Scalp",
            },

            "hierarchy":
                "Outside -> Scalp -> Skull -> CSF -> Brain/Tumor",

            "brain_envelope_definition":
                "Outer MRI head mask eroded by total "
                "CSF + Skull + Scalp shell thickness",

            "expected_interfaces": [

                "BrainTissue <-> Tumor",

                "BrainTissue <-> CSF",

                "CSF <-> Skull",

                "Skull <-> Scalp",
            ],

            "forbidden_direct_interfaces": [

                "Tumor <-> CSF",

                "Tumor <-> Skull",

                "Tumor <-> Scalp",

                "BrainTissue <-> Skull",

                "BrainTissue <-> Scalp",

                "CSF <-> Scalp",
            ],

            "outside_material":
                "Scalp",

            "input_masks_modified":
                False,

            "tumor_modified":
                False,

            "tumor_morphology":
                "disabled",
        },

        "validation": {

            "tumor_exact":
                bool(
                    tumor_original_count
                    == tumor_final_count
                ),

            "material_masks_exclusive":
                True,

            "nested_geometry":
                True,

            "expected_interfaces_only":
                True,

            "scalp_only_exterior":
                True,
        },

        "warning": (
            "CSF, Skull, and Scalp are computational "
            "morphological shells. They are not guaranteed "
            "anatomical ground-truth segmentations."
        ),
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "MRI-ONLY 5-REGION FEM PREPROCESSING"
    )

    log(
        "CORRECTED NESTED-HIERARCHY PIPELINE"
    )

    log(
        "Original MRI files: READ-ONLY"
    )

    log(
        "Original tumor segmentation: READ-ONLY"
    )

    log(
        "Tumor morphology: DISABLED"
    )

    log(
        "Brain envelope: DERIVED FROM OUTER HEAD MASK"
    )

    log(
        "Computational shells: ENABLED"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    section(
        "CHECKING INPUT FILES"
    )

    check_input_files()

    log(
        f"T1    : {T1_PATH}"
    )

    log(
        f"T1GD  : {T1GD_PATH}"
    )

    log(
        f"T2    : {T2_PATH}"
    )

    log(
        f"FLAIR : {FLAIR_PATH}"
    )

    log(
        f"MASK  : {MASK_PATH}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load MRI
    # --------------------------------------------------------

    section(
        "LOADING MRI DATA"
    )

    (
        t1,
        affine,
        spacing,
        header
    ) = load_nifti(
        T1_PATH
    )

    (
        t1gd,
        affine_t1gd,
        spacing_t1gd,
        _
    ) = load_nifti(
        T1GD_PATH
    )

    (
        t2,
        affine_t2,
        spacing_t2,
        _
    ) = load_nifti(
        T2_PATH
    )

    (
        flair,
        affine_flair,
        spacing_flair,
        _
    ) = load_nifti(
        FLAIR_PATH
    )

    (
        tumor_raw,
        affine_mask,
        spacing_mask,
        _
    ) = load_nifti(
        MASK_PATH
    )

    log(
        f"T1    : {t1.shape}"
    )

    log(
        f"T1GD  : {t1gd.shape}"
    )

    log(
        f"T2    : {t2.shape}"
    )

    log(
        f"FLAIR : {flair.shape}"
    )

    log(
        f"MASK  : {tumor_raw.shape}"
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    section(
        "CHECKING IMAGE GEOMETRY"
    )

    check_same_geometry(
        "T1",
        t1,
        affine,
        "T1GD",
        t1gd,
        affine_t1gd
    )

    check_same_geometry(
        "T1",
        t1,
        affine,
        "T2",
        t2,
        affine_t2
    )

    check_same_geometry(
        "T1",
        t1,
        affine,
        "FLAIR",
        flair,
        affine_flair
    )

    check_same_geometry(
        "T1",
        t1,
        affine,
        "MASK",
        tumor_raw,
        affine_mask
    )

    if not np.allclose(
        spacing,
        spacing_t1gd,
        atol=1e-5
    ):

        raise ValueError(
            "T1/T1GD voxel spacing mismatch."
        )

    if not np.allclose(
        spacing,
        spacing_t2,
        atol=1e-5
    ):

        raise ValueError(
            "T1/T2 voxel spacing mismatch."
        )

    if not np.allclose(
        spacing,
        spacing_flair,
        atol=1e-5
    ):

        raise ValueError(
            "T1/FLAIR voxel spacing mismatch."
        )

    if not np.allclose(
        spacing,
        spacing_mask,
        atol=1e-5
    ):

        raise ValueError(
            "T1/MASK voxel spacing mismatch."
        )

    log(
        f"Voxel spacing: "
        f"{spacing[0]:.6f} x "
        f"{spacing[1]:.6f} x "
        f"{spacing[2]:.6f} mm"
    )

    voxel_volume = (
        spacing[0]
        * spacing[1]
        * spacing[2]
    )

    log(
        f"Voxel volume: "
        f"{voxel_volume:.6f} mm^3"
    )

    # --------------------------------------------------------
    # Tumor
    # --------------------------------------------------------

    section(
        "READING ORIGINAL TUMOR SEGMENTATION"
    )

    tumor_mask = (
        np.isfinite(tumor_raw)
        &
        (tumor_raw > 0)
    )

    tumor_mask = (
        tumor_mask.astype(bool)
    )

    original_tumor_count = int(
        np.count_nonzero(
            tumor_mask
        )
    )

    log(
        f"Original tumor voxels: "
        f"{original_tumor_count:,}"
    )

    if original_tumor_count == 0:

        raise RuntimeError(
            "Tumor mask contains no positive voxels."
        )

    tumor_bbox = mask_bbox(
        tumor_mask,
        spacing,
        affine
    )

    log(
        f"Tumor voxel min: "
        f"{tumor_bbox['voxel_min']}"
    )

    log(
        f"Tumor voxel max: "
        f"{tumor_bbox['voxel_max']}"
    )

    log(
        f"Tumor world min: "
        f"{tumor_bbox['world_min_mm']}"
    )

    log(
        f"Tumor world max: "
        f"{tumor_bbox['world_max_mm']}"
    )

    log(
        "Original tumor mask will never be modified."
    )

    # --------------------------------------------------------
    # Normalize MRI
    # --------------------------------------------------------

    section(
        "NORMALIZING MRI"
    )

    t1_n = robust_normalize(
        t1
    )

    t1gd_n = robust_normalize(
        t1gd
    )

    t2_n = robust_normalize(
        t2
    )

    flair_n = robust_normalize(
        flair
    )

    # --------------------------------------------------------
    # Outer head
    # --------------------------------------------------------

    (
        head_mask,
        combined,
        head_threshold,
        otsu_threshold
    ) = estimate_head_mask(
        t1_n,
        t1gd_n,
        t2_n,
        flair_n
    )

    head_bbox = mask_bbox(
        head_mask,
        spacing,
        affine
    )

    log(
        f"Head voxels: "
        f"{np.count_nonzero(head_mask):,}"
    )

    log(
        f"Head voxel min: "
        f"{head_bbox['voxel_min']}"
    )

    log(
        f"Head voxel max: "
        f"{head_bbox['voxel_max']}"
    )

    log(
        f"Head world min: "
        f"{head_bbox['world_min_mm']}"
    )

    log(
        f"Head world max: "
        f"{head_bbox['world_max_mm']}"
    )

    brain_mask, brain_method = estimate_anatomical_brain_mask(
        combined, head_mask, tumor_mask
    )

    # --------------------------------------------------------
    # Shell parameters
    # --------------------------------------------------------

    section(
        "COMPUTATIONAL SHELL PARAMETERS"
    )

    (
        csf_voxels,
        skull_voxels,
        scalp_voxels,
        margin
    ) = calculate_domain_margin(
        spacing
    )

    total_shell_voxels = (
        csf_voxels
        + skull_voxels
        + scalp_voxels
    )

    log(
        f"CSF thickness   : "
        f"{CSF_THICKNESS_MM} mm "
        f"= {csf_voxels} voxels"
    )

    log(
        f"Skull thickness : "
        f"{SKULL_THICKNESS_MM} mm "
        f"= {skull_voxels} voxels"
    )

    log(
        f"Scalp thickness : "
        f"{SCALP_THICKNESS_MM} mm "
        f"= {scalp_voxels} voxels"
    )

    log(
        f"Total shells    : "
        f"{total_shell_voxels} voxels"
    )

    log(
        f"Extra margin    : "
        f"{EXTRA_OUTER_MARGIN_VOXELS} voxels"
    )

    log(
        f"Derived margin  : "
        f"{margin} voxels/side"
    )

    # --------------------------------------------------------
    # ORIGINAL-SIZE COMPUTATIONAL DIAGNOSTIC
    # --------------------------------------------------------

    section(
        "BUILDING ORIGINAL-SIZE COMPUTATIONAL DIAGNOSTIC"
    )

    original_regions = (
        build_nested_regions(
            head_mask,
            tumor_mask,
            csf_voxels,
            skull_voxels,
            scalp_voxels,
            brain_mask
        )
    )

    check_tumor_containment(
        tumor_mask,
        original_regions[
            "brain_envelope"
        ]
    )

    check_nested_geometry(
        original_regions
    )

    check_material_exclusivity(
        original_regions
    )

    original_material = (
        build_material_labels(
            original_regions
        )
    )

    validate_nested_material_structure(
        original_material
    )

    original_material_path = (
        OUTPUT_DIR
        / "material_labels_original.nii.gz"
    )

    save_nifti(
        original_material,
        affine,
        header,
        original_material_path,
        np.uint8
    )

    log(
        f"Original material labels written:"
    )

    log(
        f"  {original_material_path}"
    )

    # --------------------------------------------------------
    # PAD DERIVED DOMAIN
    # --------------------------------------------------------

    section(
        "BUILDING DERIVED COMPUTATIONAL DOMAIN"
    )

    (
        head_padded,
        tumor_padded,
        brain_padded,
        derived_affine
    ) = pad_derived_inputs(
        head_mask,
        tumor_mask,
        brain_mask,
        affine,
        margin
    )

    log(
        f"Derived head voxels: "
        f"{np.count_nonzero(head_padded):,}"
    )

    log(
        f"Derived tumor voxels: "
        f"{np.count_nonzero(tumor_padded):,}"
    )

    # --------------------------------------------------------
    # Exact tumor checks
    # --------------------------------------------------------

    check_tumor_preservation(
        tumor_mask,
        tumor_padded
    )

    check_embedded_tumor_exact_match(
        tumor_mask,
        tumor_padded,
        margin
    )

    # --------------------------------------------------------
    # BUILD FINAL REGIONS
    # --------------------------------------------------------

    section(
        "BUILDING FINAL NESTED FEM REGIONS"
    )

    regions = (
        build_nested_regions(
            head_padded,
            tumor_padded,
            csf_voxels,
            skull_voxels,
            scalp_voxels,
            brain_padded
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    check_tumor_containment(
        regions["Tumor"],
        regions["brain_envelope"]
    )

    check_nested_geometry(
        regions
    )

    check_material_exclusivity(
        regions
    )

    # --------------------------------------------------------
    # Material labels
    # --------------------------------------------------------

    material = (
        build_material_labels(
            regions
        )
    )

    # --------------------------------------------------------
    # Final tumor preservation
    # --------------------------------------------------------

    check_tumor_preservation(
        tumor_mask,
        regions["Tumor"]
    )

    check_embedded_tumor_exact_match(
        tumor_mask,
        regions["Tumor"],
        margin
    )

    # --------------------------------------------------------
    # Material interfaces
    # --------------------------------------------------------

    validate_nested_material_structure(
        material
    )

    # --------------------------------------------------------
    # Exterior
    # --------------------------------------------------------

    exterior_counts = (
        validate_exterior(
            material
        )
    )

    # --------------------------------------------------------
    # Strong diagnostics
    # --------------------------------------------------------

    print_region_diagnostics(
        regions,
        spacing,
        derived_affine
    )

    # --------------------------------------------------------
    # Final material histogram
    # --------------------------------------------------------

    section(
        "FINAL MATERIAL LABEL HISTOGRAM"
    )

    label_names = {
        0: "Outside",
        1: "BrainTissue",
        2: "Tumor",
        3: "CSF",
        4: "Skull",
        5: "Scalp",
    }

    for label in range(6):

        count = int(
            np.count_nonzero(
                material == label
            )
        )

        log(
            f"Label {label}: "
            f"{label_names[label]:<15} "
            f"{count:,} voxels"
        )

    # --------------------------------------------------------
    # Save final masks
    # --------------------------------------------------------

    section(
        "SAVING FINAL DERIVED FEM MASKS"
    )

    save_nifti(
        regions["BrainTissue"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "brain_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        regions["Tumor"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "tumor_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        regions["CSF"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "csf_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        regions["Skull"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "skull_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        regions["Scalp"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "scalp_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        head_padded,
        derived_affine,
        header,
        OUTPUT_DIR
        / "head_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        regions["brain_envelope"],
        derived_affine,
        header,
        OUTPUT_DIR
        / "brain_envelope_mask.nii.gz",
        np.uint8
    )

    save_nifti(
        material,
        derived_affine,
        header,
        OUTPUT_DIR
        / "material_labels.nii.gz",
        np.uint8
    )

    log(
        "All derived masks written successfully."
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    section(
        "FINAL MATERIAL STATISTICS"
    )

    names = [
        "BrainTissue",
        "Tumor",
        "CSF",
        "Skull",
        "Scalp",
    ]

    statistics = {}

    for name in names:

        stats = mask_statistics(
            regions[name],
            spacing
        )

        statistics[name] = stats

        log(
            f"{name:<15} "
            f"{stats['voxels']:>12,} voxels "
            f"{stats['volume_ml']:>12.3f} mL"
        )

    # --------------------------------------------------------
    # Interface counts
    # --------------------------------------------------------

    section(
        "FINAL MATERIAL INTERFACE COUNTS"
    )

    interface_counts = (
        count_material_interfaces(
            material
        )
    )

    interface_names = {
        (1, 2):
            "BrainTissue <-> Tumor",

        (1, 3):
            "BrainTissue <-> CSF",

        (3, 4):
            "CSF <-> Skull",

        (4, 5):
            "Skull <-> Scalp",
    }

    for pair, name in (
        interface_names.items()
    ):

        count = interface_counts.get(
            pair,
            0
        )

        log(
            f"{name:<30} "
            f"{count:,} faces"
        )

    # --------------------------------------------------------
    # Final tumor exact check
    # --------------------------------------------------------

    tumor_final_count = int(
        np.count_nonzero(
            regions["Tumor"]
        )
    )

    tumor_exact = bool(
        tumor_final_count
        == original_tumor_count
    )

    if not tumor_exact:

        raise RuntimeError(
            "FINAL TUMOR PRESERVATION FAILED."
        )

    # --------------------------------------------------------
    # Final exterior assertion
    # --------------------------------------------------------

    if any(
        exterior_counts[label] != 0
        for label in [1, 2, 3, 4]
    ):

        raise RuntimeError(
            "FINAL EXTERIOR VALIDATION FAILED."
        )

    if exterior_counts[5] == 0:

        raise RuntimeError(
            "FINAL EXTERIOR VALIDATION FAILED: "
            "Scalp does not touch outside."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        original_shape=t1.shape,
        final_shape=material.shape,
        spacing=spacing,
        margin=margin,
        csf_voxels=csf_voxels,
        skull_voxels=skull_voxels,
        scalp_voxels=scalp_voxels,
        head_threshold=head_threshold,
        otsu_threshold=otsu_threshold,
        original_tumor=tumor_mask,
        derived_tumor=regions["Tumor"],
        material=material,
        original_material=original_material,
        exterior_counts=exterior_counts,
        original_affine=affine,
        derived_affine=derived_affine,
        regions=regions
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=json_default
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    section(
        "PREPROCESSING COMPLETE"
    )

    print(
        f"Original shape : {t1.shape}"
    )

    print(
        f"Derived shape  : {material.shape}"
    )

    print(
        f"Derived margin : {margin} voxels/side"
    )

    print()

    print(
        "COMPUTATIONAL HIERARCHY:"
    )

    print(
        "  Outside"
    )

    print(
        "    -> Scalp"
    )

    print(
        "      -> Skull"
    )

    print(
        "        -> CSF"
    )

    print(
        "          -> BrainTissue"
    )

    print(
        "             -> Tumor"
    )

    print()

    print(
        f"Original tumor : "
        f"{original_tumor_count:,}"
    )

    print(
        f"Final tumor    : "
        f"{tumor_final_count:,}"
    )

    print(
        f"Tumor exact    : "
        f"{tumor_exact}"
    )

    print()

    print(
        "FINAL MATERIALS:"
    )

    for name in names:

        count = int(
            np.count_nonzero(
                regions[name]
            )
        )

        print(
            f"  {name:<15}: "
            f"{count:,} voxels"
        )

    print()

    print(
        "EXTERIOR CONTACT:"
    )

    print(
        f"  BrainTissue -> outside: "
        f"{exterior_counts[1]:,}"
    )

    print(
        f"  Tumor       -> outside: "
        f"{exterior_counts[2]:,}"
    )

    print(
        f"  CSF         -> outside: "
        f"{exterior_counts[3]:,}"
    )

    print(
        f"  Skull       -> outside: "
        f"{exterior_counts[4]:,}"
    )

    print(
        f"  Scalp       -> outside: "
        f"{exterior_counts[5]:,}"
    )

    print()

    print(
        "OUTPUT:"
    )

    print(
        f"  Masks  : {OUTPUT_DIR}"
    )

    print(
        f"  Report : {REPORT_PATH}"
    )

    print()

    print(
        "VALIDATION:"
    )

    print(
        "  [PASS] Original MRI unchanged"
    )

    print(
        "  [PASS] Original tumor unchanged"
    )

    print(
        "  [PASS] Tumor voxel count preserved"
    )

    print(
        "  [PASS] Tumor embedded mask exact match"
    )

    print(
        "  [PASS] Tumor contained in brain envelope"
    )

    print(
        "  [PASS] Material masks mutually exclusive"
    )

    print(
        "  [PASS] Nested geometry"
    )

    print(
        "  [PASS] Expected material interfaces"
    )

    print(
        "  [PASS] Scalp-only exterior"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The original MRI files were not modified."
    )

    print(
        "The original tumor segmentation was not modified."
    )

    print(
        "No morphology was applied to MASK.nii.gz."
    )

    print(
        "The brain envelope is derived from the outer MRI head."
    )

    print(
        "CSF, Skull, and Scalp are computational shells."
    )

    print(
        "Only Scalp is permitted to touch computational outside."
    )

    print("=" * 88)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 88)
        print("PREPROCESSING FAILED")
        print("=" * 88)
        print(str(exc))
        print("=" * 88)

        sys.exit(1)
