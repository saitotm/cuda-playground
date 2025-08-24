TARGET = vec_add.out
SRC = vec_add.cu

NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++20
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)