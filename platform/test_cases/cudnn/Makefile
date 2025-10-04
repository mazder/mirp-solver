# Makefile for CUDA + cuDNN project

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudnn -lcuda

# Source and output files
SRC = test_cudnn.cpp
OUT = test_cudnn

# Target to compile the project
all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

# Clean target to remove generated files
clean:
	rm -f $(OUT)

# Run the compiled program
run: $(OUT)
	./$(OUT)
