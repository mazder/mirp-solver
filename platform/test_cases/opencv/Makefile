# Makefile for compiling and linking OpenCV with CUDA support

# Define the compiler and flags
CXX = g++
CXXFLAGS = -std=c++11

# Directories
INCDIR = /usr/local/include/opencv4
LIBDIR = /usr/local/lib

# OpenCV Libraries (including specific CUDA libraries)
OPENCV_LIBS = -lopencv_core -lopencv_cudaimgproc -lopencv_imgproc -lopencv_highgui -lopencv_cudacodec -lopencv_cudafilters -lopencv_cudalegacy `pkg-config --cflags --libs opencv4`

# Output binary
TARGET = test_opencv_gpu

# Source files
SRC = test_opencv_gpu.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Build the program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(OPENCV_LIBS) -L$(LIBDIR)

# Compile the source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

# Install the program (optional)
install:
	cp $(TARGET) /usr/local/bin/

.PHONY: all clean install

