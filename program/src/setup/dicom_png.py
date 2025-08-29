# dicom_png.py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

def run_dicom_png(dicom_file):
    """Convert a DICOM file into PNG and save it."""

    try:
        dataset = pydicom.dcmread(dicom_file)
    except Exception as e:
        print(f"[ERROR] Failed to load DICOM {dicom_file}: {e}")
        return None

    if "PixelData" not in dataset:
        print("[ERROR] No pixel data found in DICOM.")
        return None

    image = dataset.pixel_array.astype(float)

    # Normalize image to 0-255
    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = np.uint8(image)

    # Output file
    out_file = dicom_file + "_converted.png"
    plt.imsave(out_file, image, cmap="gray")

    print(f"[INFO] Saved PNG: {out_file}")
    return out_file
