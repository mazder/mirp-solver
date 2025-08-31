#dataset_splitter.py

import os
import shutil
import random
from PIL import Image, ImageOps


class DatasetSplitter:
    def __init__(self, base_dir, output_dir, patients=None, n_folds=5, val_split=0.2, test_split=0.1, seed=42):
        """
        Prepare dataset folds with optional augmentation.
        :param base_dir: Directory containing patient folders with PNG images
        :param output_dir: Directory where folds will be saved
        :param patients: List of patient names to include (e.g., ["PAT2","PAT6"])
        :param n_folds: Number of folds for cross-validation
        :param val_split: Fraction of training set used as validation
        :param test_split: Fraction of total set used as test
        :param seed: Random seed
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.val_split = val_split
        self.test_split = test_split
        self.patients = patients or [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        random.seed(seed)

    def augment_image(self, img: Image.Image):
        """Apply simple augmentations to the image (flip, rotation)."""
        if random.random() > 0.5:
            img = ImageOps.mirror(img)
        if random.random() > 0.5:
            img = ImageOps.flip(img)
        rotations = [0, 90, 180, 270]
        img = img.rotate(random.choice(rotations))
        return img

    def create_folds(self):
        """Create folds with train/val/test splits and optional augmentation."""
        random.shuffle(self.patients)

        # --- ensure at least 1 patient in test ---
        n_test = max(1, int(len(self.patients) * self.test_split))
        test_patients = self.patients[:n_test]
        train_val_patients = self.patients[n_test:]

        for fold in range(self.n_folds):
            fold_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            train_dir = os.path.join(fold_dir, "train")
            val_dir = os.path.join(fold_dir, "val")
            test_dir = os.path.join(fold_dir, "test")

            # Clean directories if they exist
            for d in [train_dir, val_dir, test_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

            # Split train/val patients
            random.shuffle(train_val_patients)
            n_val = max(1, int(len(train_val_patients) * self.val_split)) if train_val_patients else 0
            val_patients = train_val_patients[:n_val]
            train_patients = train_val_patients[n_val:]

            # Copy PNG sequences for each dataset
            for dataset_dir, patient_list in zip([train_dir, val_dir, test_dir],
                                                 [train_patients, val_patients, test_patients]):
                for patient in patient_list:
                    patient_src_dir = os.path.join(self.base_dir, patient)
                    patient_dest_dir = os.path.join(dataset_dir, patient)
                    os.makedirs(patient_dest_dir, exist_ok=True)

                    # Find all sequence PNG folders (ending with _PNG)
                    for seq_folder in os.listdir(patient_src_dir):
                        if seq_folder.endswith("_PNG"):
                            seq_src_path = os.path.join(patient_src_dir, seq_folder)
                            seq_dest_path = os.path.join(patient_dest_dir, seq_folder)
                            os.makedirs(seq_dest_path, exist_ok=True)

                            for img_file in os.listdir(seq_src_path):
                                if img_file.endswith(".png"):
                                    img_path = os.path.join(seq_src_path, img_file)
                                    img = Image.open(img_path)
                                    if dataset_dir == train_dir:
                                        img = self.augment_image(img)
                                    img.save(os.path.join(seq_dest_path, img_file))

            print(f"[INFO] Fold {fold+1} created with "
                  f"{len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")

    def create_round_robin_folds(self):
        """
        Create folds with round-robin patient assignment:
        - Each patient appears in the test set exactly once across folds.
        - Remaining patients are split into train/val.
        """
        patients = list(self.patients)
        random.shuffle(patients)  # shuffle once to avoid bias

        for fold in range(self.n_folds):
            fold_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            train_dir = os.path.join(fold_dir, "train")
            val_dir = os.path.join(fold_dir, "val")
            test_dir = os.path.join(fold_dir, "test")

            # Clean and recreate directories
            for d in [train_dir, val_dir, test_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

            # --- round robin: assign 1 patient to test in this fold ---
            test_patients = [patients[fold % len(patients)]]
            train_val_patients = [p for p in patients if p not in test_patients]

            # --- validation split from remaining ---
            random.shuffle(train_val_patients)
            n_val = max(1, int(len(train_val_patients) * self.val_split)) if train_val_patients else 0
            val_patients = train_val_patients[:n_val]
            train_patients = train_val_patients[n_val:]

            # Copy data
            for dataset_dir, patient_list in zip([train_dir, val_dir, test_dir],
                                                 [train_patients, val_patients, test_patients]):
                for patient in patient_list:
                    patient_src_dir = os.path.join(self.base_dir, patient)
                    patient_dest_dir = os.path.join(dataset_dir, patient)
                    os.makedirs(patient_dest_dir, exist_ok=True)

                    for seq_folder in os.listdir(patient_src_dir):
                        if seq_folder.endswith("_PNG"):
                            seq_src_path = os.path.join(patient_src_dir, seq_folder)
                            seq_dest_path = os.path.join(patient_dest_dir, seq_folder)
                            os.makedirs(seq_dest_path, exist_ok=True)

                            for img_file in os.listdir(seq_src_path):
                                if img_file.endswith(".png"):
                                    img_path = os.path.join(seq_src_path, img_file)
                                    img = Image.open(img_path)
                                    if dataset_dir == train_dir:
                                        img = self.augment_image(img)
                                    img.save(os.path.join(seq_dest_path, img_file))

            print(f"[INFO] Fold {fold+1} created (Round-Robin) with "
                  f"{len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")


## ========== CLI ==========
#if __name__ == "__main__":
#    if len(sys.argv) < 2:
#        print("Usage: python dicom_png.py dicompng <base_dir>")
#        sys.exit(1)
#    base_dir = "/workspace/mirp-solver/simulation_data/dicom"
#    output_dir = "/workspace/mirp-solver/simulation_data/folds"
#    patients = ["PAT2", "PAT6", "PAT15", "PAT24"]
#
#    # Create and run splitter
#    splitter = DatasetSplitter(
#        base_dir=base_dir,
#        output_dir=output_dir,
#        patients=patients,
#        n_folds=5,
#        val_split=0.2,
#        test_split=0.1,
#        seed=42
#    )
#    splitter.create_folds()
