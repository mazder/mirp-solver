# Define the CUDA compiler
NVCC = nvcc

# Define the output executable
TARGET = test_cuda

# Define source files
SRC = test_cuda.cu

# Define compilation flags
CFLAGS = -O3  # Optimization flag (you can change it as needed)
LDFLAGS =  # Linker flags (add any necessary libraries)

# Default target to build the project
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean the generated files
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)
