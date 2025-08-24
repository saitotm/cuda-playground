TARGETS = vec_add.out vec_add_float4.out

NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++20
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80

all: $(TARGETS)

vec_add.out: vec_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vec_add_float4.out: vec_add_float4.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean