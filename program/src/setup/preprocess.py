# preprocess.py
from dicom_png import run_dicom_png

class Preprocess:
    def __init__(self, dicom_file: str):
        """
        Initialize preprocessing with the given DICOM file path.
        """
        self.dicom_file = dicom_file

    def dicom_to_png(self):
        run_dicom_png(self.dicom_file)
