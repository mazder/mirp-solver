#!/bin/bash

cd cuda
mkdir -p build
cd build
rm -rf *
cmake .. -Wno-dev
make
./test_cuda
cd ..
rm -rf build
cd ..

cd cudnn
mkdir -p build
cd build
rm -rf *
cmake .. -Wno-dev
make
./test_cudnn
cd ..
rm -rf build
cd ..

cd opencv
python3 test.py
make
./test_opencv_gpu
rm -rf output_image.jpg
make clean
cd ..

cd pytorch
mkdir -p build
cd build
rm -rf *
cmake .. -Wno-dev
make
./main_app
cd ..
rm -rf build
cd ..

cd vtk
python3 vtk_test.py
rm -rf __pycache__
mkdir -p build
cd build
rm -rf *
cmake .. -Wno-dev
make
./main
ls sphere.vtp
cd ..
rm -rf build
cd ..

cd mpi
mkdir -p build
cd build
rm -rf *
cmake .. -Wno-dev
make
mpirun -np 2 ./mpi_program
cd ..
rm -rf build
cd ..

export PATH=/opt/gcc-14/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc-14/lib64:$LD_LIBRARY_PATH
python3 -c 'import nibabel, vtk, vtkbone, pydicom, PIL, numpy; print("all good!")'

source /opt/venv_unet/bin/activate
nnUNetv2_train --help
deactivate
