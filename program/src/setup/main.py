# main.py
import sys
from preprocess import Preprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <dicom_file>")
        sys.exit(1)

    dicom_file = sys.argv[1]
    proc_obj = Preprocess(dicom_file)
    proc_obj.dicom_to_png()
    print(f"[INFO] Saved result to: {dicom_file}_converted.png")

if __name__ == "__main__":
    main()
