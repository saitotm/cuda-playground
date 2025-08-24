TARGETS = vec_add.out vec_add_float4.out
PTX_FILES = vec_add.ptx vec_add_float4.ptx

NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++20
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80

all: $(TARGETS)

# Executable targets
vec_add.out: vec_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vec_add_float4.out: vec_add_float4.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# PTX generation targets
ptx: $(PTX_FILES)

vec_add.ptx: vec_add.cu
	$(NVCC) $(NVCC_FLAGS) -ptx -o $@ $<

vec_add_float4.ptx: vec_add_float4.cu
	$(NVCC) $(NVCC_FLAGS) -ptx -o $@ $<

# Clean up
clean:
	rm -f $(TARGETS) $(PTX_FILES)

.PHONY: all clean ptx