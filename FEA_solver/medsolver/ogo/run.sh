#!/bin/bash

export PATH=/opt/gcc-14/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc-14/lib64:$LD_LIBRARY_PATH
python3 -c 'import nibabel, vtk, vtkbone, pydicom, PIL, numpy; print("all good!")'

python3 apply_calibration.py
