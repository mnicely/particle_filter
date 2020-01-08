CUDA_V		:=10.2
CPP			:=g++
NVCC		:=/usr/local/cuda-$(CUDA_V)/bin/nvcc
INC			:=-I/usr/local/cuda/samples/common/inc
LIBS 		:=-llapacke -lopenblas -lboost_program_options
ARCHES		:=-gencode arch=compute_70,code=\"compute_70,sm_70\"
SRCDIR		:=src
OBJDIR		:=obj
APP			:=filters
CFLAGS		:=-std=c++14  -O3 --use_fast_math
NVTX		:=

ifdef NVTX # NVTX labeling/profiling
	LIBS		+= -lnvToolsExt
	CFLAGS		+= -DUSE_NVTX
endif

# Add inputs and outputs from these tool invocations to the build variables 
SOURCES := 	command_line_options \
			filters \
			generate_data \
			particle_bpf_cpu \
			particle_bpf_gpu

OBJECTS		+=$(addprefix $(OBJDIR)/, $(SOURCES:%=%.o))

# All Target
all: build $(APP)

build:	
	@mkdir -p $(OBJDIR)

$(APP): $(OBJECTS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	$(NVCC) --cudart=static -ccbin $(CPP) $(ARCHES) -o $@ $(OBJECTS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJDIR)/%.o: ./$(SRCDIR)/%.cpp
	$(NVCC) -x cu $(INC) -ccbin $(CPP) $(CFLAGS) $(ARCHES) -c -o "$@" "$<" --expt-relaxed-constexpr

$(OBJDIR)/%.o: ./$(SRCDIR)/%.cu
	$(NVCC) $(INC) -ccbin $(CPP) $(CFLAGS) $(ARCHES) -c -o "$@" "$<" --expt-relaxed-constexpr

clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(OBJDIR)/*.o $(APP)'
	@echo ' '
	@rm -rf $(OBJDIR)/*.o $(APP)

.PHONY: all build clean 
