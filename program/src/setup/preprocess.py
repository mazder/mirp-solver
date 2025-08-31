# preprocess.py
import sys
import os
from dicom_png import DicomPng
from dataset_splitter import DatasetSplitter

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py [dicompng|folds] <base_path>")
        sys.exit(1)

    mode = sys.argv[1]
    base_path = sys.argv[2]  # e.g. /workspace/mirp-solver
    patients = ["PAT2", "PAT6", "PAT15", "PAT24"]

    if mode == "dicompng":
        print("[INFO] Running DICOM to PNG conversion ...")
        dicom_root = os.path.join(base_path, "simulation_data", "dicom")
        dp = DicomPng(base_dir=dicom_root, patients=patients)
        dp.dicom_to_png()

    elif mode == "pngfolds":
        dicom_root = os.path.join(base_path, "simulation_data", "dicom")
        output_dir = os.path.join(base_path, "simulation_data", "folds")
        splitter = DatasetSplitter(
            base_dir=dicom_root,
            output_dir=output_dir,
            patients=patients,
            n_folds=5,
            val_split=0.2,
            test_split=0.1,
            seed=42
        )
        splitter.create_folds()
    elif mode == "pngfolds_rr":
        dicom_root = os.path.join(base_path, "simulation_data", "dicom")
        output_dir = os.path.join(base_path, "simulation_data", "folds_round_robin")
        splitter = DatasetSplitter(
            base_dir=dicom_root,
            output_dir=output_dir,
            patients=patients,
            n_folds=5,
            val_split=0.2,
            test_split=0.1,  # ignored in round robin
            seed=42
        )
        splitter.create_round_robin_folds()

    else:
        print(f"[ERROR] Unknown mode: {mode}")
