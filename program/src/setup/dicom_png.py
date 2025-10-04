# dicom_png.py

import os
import sys
import numpy as np
import nibabel as nib
from pydicom.filereader import dcmread
from PIL import Image
import shutil


class DicomPng:
    def __init__(self, base_dir: str, patients=None):
        """
        :param base_dir: Root directory containing patient DICOM folders
        :param patients: List of patient folder names (if None, auto-detects)
        """
        self.base_dir = base_dir
        self.patients = patients

    def dicom_to_png(self):
        """Convert all DICOM → NIfTI → PNG for all patients."""
        for patient in self.patients:
            patient_dir = os.path.join(self.base_dir, patient)
            if not os.path.exists(patient_dir):
                print(f"[WARNING] Patient directory does not exist: {patient_dir}")
                continue

            for seq_name in os.listdir(patient_dir):
                seq_path = os.path.join(patient_dir, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                png_output_dir = seq_path + "_PNG"
                nifti_output_file = seq_path + ".nii"

                # Clean existing outputs
                if os.path.exists(png_output_dir):
                    print(f"[INFO] Removing existing PNG folder: {png_output_dir}")
                    shutil.rmtree(png_output_dir)

                if os.path.exists(nifti_output_file):
                    print(f"[INFO] Removing existing NIfTI file: {nifti_output_file}")
                    os.remove(nifti_output_file)

                print(f"[INFO] Processing DICOM sequence: {seq_path}")
                try:
                    self.run_dicom_to_png(seq_path, png_output_dir, nifti_output_file)
                except Exception as e:
                    print(f"[ERROR] Failed to process {seq_path}: {e}")

    def dicom_to_nifti(self, dicom_dir: str, output_file: str):
        """Convert a folder of DICOM slices into a single NIfTI file."""
        dicom_files = [os.path.join(dicom_dir, f)
                       for f in os.listdir(dicom_dir)
                       if f.endswith(".dcm")]
        dicom_files.sort()

        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        slices = [dcmread(f).pixel_array for f in dicom_files]
        volume = np.stack(slices, axis=-1).astype(np.float32)

        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(nifti_img, output_file)
        print(f"[INFO] DICOM → NIfTI saved: {output_file}")
        return output_file

    def nifti_to_png(self, nifti_file: str, output_dir: str):
        """Convert a NIfTI file into PNG slices."""
        os.makedirs(output_dir, exist_ok=True)

        nii_img = nib.load(nifti_file)
        image_data = np.asarray(nii_img.get_fdata(), dtype=np.float32)

        n_slices = image_data.shape[2]
        for slice_idx in range(n_slices):
            cur_slice = image_data[:, :, slice_idx]
            cur_slice = np.rot90(cur_slice)

            if cur_slice.max() > 0:
                cur_slice = cur_slice / cur_slice.max()
            cur_slice_uint8 = (cur_slice * 255).astype(np.uint8)

            png_filename = os.path.join(output_dir, f"slice{slice_idx+1}.png")
            Image.fromarray(cur_slice_uint8).save(png_filename)

        print(f"[INFO] NIfTI → PNG saved in: {output_dir}")

    def run_dicom_to_png(self, dicom_input_dir: str, png_output_dir: str, nifti_output_file: str):
        """Full pipeline: DICOM → NIfTI → PNG"""
        print(f"[INFO] Converting {dicom_input_dir} → {png_output_dir}")
        nifti_file = self.dicom_to_nifti(dicom_input_dir, nifti_output_file)
        self.nifti_to_png(nifti_file, png_output_dir)
        print("[INFO] Pipeline completed successfully!")


## ========== CLI ==========
#if __name__ == "__main__":
#    if len(sys.argv) < 2:
#        print("Usage: python dicom_png.py dicompng <base_dir>")
#        sys.exit(1)
#
#    if sys.argv[1] == "dicompng":
#        base_dir = "/workspace/mirp-solver/simulation_data/dicom"
#        patients = ["PAT2", "PAT6", "PAT15", "PAT24"]
#        dp = DicomPng(base_dir=base_dir, patients=patients)
#        dp.dicom_to_png()
