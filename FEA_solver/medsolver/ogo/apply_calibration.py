import os
import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from ogo.cli.ImageCalibration import internal

matplotlib.use('Agg')  # Use non-interactive backend


class CTCalibration:
    def __init__(self, input_image, calib_txt, output_image):
        self.input_image = input_image
        self.calib_txt = calib_txt
        self.output_image = output_image

        if not os.path.exists(self.input_image):
            raise FileNotFoundError(f"Input image not found: {self.input_image}")
        if not os.path.exists(self.calib_txt):
            raise FileNotFoundError(f"Calibration file not found: {self.calib_txt}")

    def quick_check(self):
        """Check voxel distributions in CT image."""
        ct = sitk.ReadImage(self.input_image)
        ct_array = sitk.GetArrayFromImage(ct)

        air_voxels = np.sum(ct_array < -500)
        fat_voxels = np.sum((ct_array >= -190) & (ct_array <= -50))
        muscle_voxels = np.sum((ct_array >= 30) & (ct_array <= 70))

        print(f"Air voxels: {air_voxels}")
        print(f"Fat voxels: {fat_voxels}")
        print(f"Muscle voxels: {muscle_voxels}")

    def inspect(self):
        """Inspect unique labels in the image/mask."""
        mask = sitk.ReadImage(self.input_image)
        mask_array = sitk.GetArrayFromImage(mask)
        print("Unique labels in mask:", np.unique(mask_array))

    def apply_internal_calibration(self):
        """Apply calibration using OGO internal calibration tool, then visualize."""
        internal(
            input_image=self.input_image,
            input_mask=self.calib_txt,  # .txt mask triggers reapplication
            output_image=self.output_image,
            calib_file_name=None,
            useLabels=[],
            corticalbone=1,
            tissue_checks=True,
            output_arch_image=True,
            overwrite=True,
            func=None,
        )

        raw = sitk.ReadImage(self.input_image)
        cal = sitk.ReadImage(self.output_image)

        raw_arr = sitk.GetArrayFromImage(raw)
        cal_arr = sitk.GetArrayFromImage(cal)

        mid = raw_arr.shape[0] // 2
        raw_slice = raw_arr[mid, :, :]
        cal_slice = cal_arr[mid, :, :]

        raw_vmin, raw_vmax = np.percentile(raw_slice, [1, 99])
        cal_vmin, cal_vmax = np.percentile(cal_slice[cal_slice > 0], [1, 99])

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Raw CT
        im0 = axs[0].imshow(raw_slice, cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
        axs[0].set_title('Raw CT (HU)')
        axs[0].axis('off')
        fig.colorbar(im0, ax=axs[0], shrink=0.8).set_label('HU')

        # Calibrated Density
        im1 = axs[1].imshow(cal_slice, cmap='inferno', vmin=cal_vmin, vmax=cal_vmax)
        axs[1].set_title('Calibrated Density (mg/cc)')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], shrink=0.8).set_label('mg/cc')

        plt.tight_layout()
        comp_path = os.path.join(os.path.dirname(self.output_image), "ct_calibration_comparison.png")
        plt.savefig(comp_path, dpi=300)
        print("Comparison plot saved to:", comp_path)

        # Histogram
        raw_vals = raw_arr.flatten()
        cal_vals = cal_arr.flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(raw_vals, bins=100, alpha=0.6, label="Raw CT (HU)", color="gray")
        plt.hist(cal_vals, bins=100, alpha=0.6, label="Calibrated (mg/cc)", color="orange")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Histogram of Voxel Intensities")
        plt.grid(True)
        plt.tight_layout()
        hist_path = os.path.join(os.path.dirname(self.output_image), "ct_calibration_histogram.png")
        plt.savefig(hist_path, dpi=300)
        print("Histogram saved to:", hist_path)


#if __name__ == "__main__":
#    # Example usage
#    input_image = "sub-verse004_ct.nii.gz"
#    calib_txt = "sub-verse004_0000.txt"
#    output_image = "sub-verse004_ct_calibrated.nii.gz"
#
#    calib = CTCalibration(input_image, calib_txt, output_image)
#    calib.quick_check()
#    calib.inspect()
#    calib.apply_internal_calibration()

