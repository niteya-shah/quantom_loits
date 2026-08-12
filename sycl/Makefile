include ../makefile_defs.mk

#enable the following for NVIDIA NSight profiling:
#CXXFLAGS+= -DPROFILING=1 -I/opt/nvidia/nsight-compute/2024.1.0/host/target-linux-x64/nvtx/include
WRAPPER_NAME:=$(shell python3-config --extension-suffix)
DPCPP_CANON_WRAPPER := dpcp/sycl_wrapper$(WRAPPER_NAME)
DPCPP_TBB_WRAPPER   := dpcp/sycl_wrapper-tbb$(WRAPPER_NAME)
DPCPP_CUDA_WRAPPER  := dpcp/sycl_wrapper-cud$(WRAPPER_NAME)
DPCPP_HIP_WRAPPER   := dpcp/sycl_wrapper-hip$(WRAPPER_NAME)
DPCPP_XPU_WRAPPER   := dpcp/sycl_wrapper-xpu$(WRAPPER_NAME)

.PHONY: activate_dpcpp_tbb activate_dpcpp_cuda activate_dpcpp_hip activate_dpcpp_xpu

ACPP_LIB_PATH:=$(shell echo $(ADAPTIVECPP_INSTALL_ROOT)/lib)
DPCPP_HIP_LIBSPIRV_BC := $(shell find $(DPCPPHIP_INSTALL_ROOT) -type f -name 'libspirv*.bc' 2>/dev/null | head -n 1)
IMPLEMENTATION_FLAGS:=-DSYCL_BUILD=1 -DWRAPPER_BUILD=1
all: create_binary_directories acpp/sycl_wrapper$(WRAPPER_NAME)

dpc++_for_tbb: $(DPCPP_TBB_WRAPPER) activate_dpcpp_tbb
dpc++_for_cuda: $(DPCPP_CUDA_WRAPPER) activate_dpcpp_cuda
dpc++_for_hip: $(DPCPP_HIP_WRAPPER) activate_dpcpp_hip
dpc++_for_spirv: $(DPCPP_XPU_WRAPPER) activate_dpcpp_xpu

create_binary_directories:
	mkdir -p acpp dpcp usy

###############################################################################
#                                  Sampler                                    #
###############################################################################
libsampler-acpp-gen.so: sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(ADAPTIVECPP) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS) -O3 -std=c++17 --acpp-targets="generic" -ffast-math -shared -ldl -fPIC -o $@ $^

acpp/sycl_wrapper$(WRAPPER_NAME): libsampler-acpp-gen.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS) \
		-L$(ACPP_LIB_PATH) -Wl,-rpath,$(ACPP_LIB_PATH) -lacpp-rt -lacpp-common \
		-L. -Wl,-rpath,. $^ \
		-ltorch -ldl \
		libsampler-acpp-gen.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)
		#-fsanitize=address -shared-libasan \

libsampler-dpcp-tbb.so: sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(DPCPPCPU) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS) -O3 -std=c++17 \
		-fsycl -fsycl-targets=native_cpu \
		-DCPU_SELECTOR=1\
		-Wl,-rpath=$(DPCPPCPU_INSTALL_ROOT)/lib -L$(DPCPPCPU_INSTALL_ROOT)/lib \
		-lsycl \
		-ffast-math -march=native -shared -ldl -fPIC -lz \
		-o $@ $^

$(DPCPP_TBB_WRAPPER): libsampler-dpcp-tbb.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		-march=native \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. libsampler-dpcp-tbb.so pybind_wrapper.cpp \
		-ltorch -ldl -lz \
		libsampler-dpcp-tbb.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

libsampler-dpcp-cud.so: sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(DPCPPCUDA) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS) -DGPU_SELECTOR=1 -O3 -std=c++17 \
		-fsycl -fsycl-targets=nvptx64-nvidia-cuda \
		-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=$(CUDA_DEV_TARGET) \
		-Wl,-rpath=$(DPCPPCUDA_INSTALL_ROOT)/lib -L$(DPCPPCUDA_INSTALL_ROOT)/lib \
		-lsycl -Wl,-rpath=$(CUDA_PATH)/extras/CUPTI/lib64 -lur_loader \
		-ffast-math -march=native -shared -ldl -fPIC -lz \
		-o $@ $^

$(DPCPP_CUDA_WRAPPER): libsampler-dpcp-cud.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		-march=native \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. libsampler-dpcp-cud.so pybind_wrapper.cpp \
		-ltorch -ldl -lz \
		libsampler-dpcp-cud.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

libsampler-dpcp-hip.so: sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	@test -n "$(DPCPP_HIP_LIBSPIRV_BC)" || (echo "Could not locate libspirv*.bc under $(DPCPPHIP_INSTALL_ROOT)" >&2; exit 1)
	$(DPCPPHIP) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS)  -DGPU_SELECTOR=1 -O3 -std=c++17 \
		-fsycl -fsycl-targets=amdgcn-amd-amdhsa \
		-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_DEV_TARGET) \
		-fsycl-libspirv-path=$(DPCPP_HIP_LIBSPIRV_BC) \
		-Wl,-rpath=$(DPCPPHIP_INSTALL_ROOT)/lib -L$(DPCPPHIP_INSTALL_ROOT)/lib \
		-lsycl -lur_loader -ffast-math -march=native -shared -ldl -fPIC -lz \
		-o $@ $^

$(DPCPP_HIP_WRAPPER): libsampler-dpcp-hip.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		-march=native \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. libsampler-dpcp-hip.so pybind_wrapper.cpp \
		-ltorch -ldl -lz \
		libsampler-dpcp-hip.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

libsampler-dpcp-spirv.so: sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(ICPX) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS) -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice -Wl,-rpath=$(ONEAPI_INSTALL_ROOT)/lib -L$(ONEAPI_INSTALL_ROOT)/lib -lsycl -lur_loader -ffast-math -march=native -shared -ldl -fPIC -lz -o $@ $^

$(DPCPP_XPU_WRAPPER): libsampler-dpcp-spv.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		-march=native \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. libsampler-dpcp-spv.so pybind_wrapper.cpp \
		-ltorch -ldl -lz \
		libsampler-dpcp-spv.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

libsampler-usy.so: sycl_sampler.cpp
	$(UNISYCL) $(CXXFLAGS) $(IMPLEMENTATION_FLAGS) -O3 -shared -std=c++11 -fPIC -o $@ $^

usy/sycl_wrapper$(WRAPPER_NAME): libsampler-usy.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		-march=native \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. $^ \
		-ltorch -ldl -lz	\
		libsampler-usy.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

activate_dpcpp_tbb: $(DPCPP_TBB_WRAPPER)
	cp -f $(DPCPP_TBB_WRAPPER) $(DPCPP_CANON_WRAPPER)

activate_dpcpp_cuda: $(DPCPP_CUDA_WRAPPER)
	cp -f $(DPCPP_CUDA_WRAPPER) $(DPCPP_CANON_WRAPPER)

activate_dpcpp_hip: $(DPCPP_HIP_WRAPPER)
	cp -f $(DPCPP_HIP_WRAPPER) $(DPCPP_CANON_WRAPPER)

activate_dpcpp_xpu: $(DPCPP_XPU_WRAPPER)
	cp -f $(DPCPP_XPU_WRAPPER) $(DPCPP_CANON_WRAPPER)

###############################################################################
#                                   Tests                                     #
###############################################################################
# UniSYCL
sycl_test_usy: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
		$(UNISYCL) $(IMPLEMENTATION_FLAGS) -save-kernels $(CXXFLAGS) -std=c++17 -g -O3 -o $@ $^
		#$(UNISYCL) $(CXXFLAGS) -std=c++17 -g -O3 -o $@ $^

# DPC++
sycl_test_dpcpp_hip: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	@test -n "$(DPCPP_HIP_LIBSPIRV_BC)" || (echo "Could not locate libspirv*.bc under $(DPCPPHIP_INSTALL_ROOT)" >&2; exit 1)
	$(DPCPPHIP) $(IMPLEMENTATION_FLAGS) -std=c++17 \
		-fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 \
		-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_DEV_TARGET) \
		-fsycl-libspirv-path=$(DPCPP_HIP_LIBSPIRV_BC) \
		$(CXXFLAGS) \
		-Wl,-rpath=$(DPCPPHIP_INSTALL_ROOT)/lib -L$(DPCPPHIP_INSTALL_ROOT)/lib \
		-lsycl -o $@ $^

sycl_test_dpcpp_cuda: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(DPCPPCUDA) $(IMPLEMENTATION_FLAGS) -O3 -std=c++17 \
		-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 \
		-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=$(CUDA_DEV_TARGET) \
		$(CXXFLAGS) \
		-Wl,-rpath=$(DPCPPCUDA_INSTALL_ROOT)/lib -L$(DPCPPCUDA_INSTALL_ROOT)/lib \
		-lsycl -Wl,-rpath=$(CUDA_PATH)/extras/CUPTI/lib64 -lur_loader -lz \
		-o $@ $^

# AdaptiveCpp
sycl_test_acpp: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(ADAPTIVECPP) $(IMPLEMENTATION_FLAGS) -std=c++17 --acpp-targets="generic" -O0 -fno-omit-frame-pointer -g $(CXXFLAGS) -fPIC -o $@ $^

sycl_test_acpp_omp: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(ADAPTIVECPP) $(IMPLEMENTATION_FLAGS) -std=c++17 --acpp-targets="omp" -O3 -g $(CXXFLAGS) -fPIC -o $@ $^

sycl_test_acpp_cud: sycl_test.cpp sycl_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(ADAPTIVECPP) $(IMPLEMENTATION_FLAGS) -std=c++17 --acpp-targets="cuda.explicit-multipass:$(CUDA_DEV_TARGET)" -I$(NVMATHLIBS)/include -L$(NVMATHLIBS)/lib64 -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib -O3 --acpp-nvcxx=$(NVCXX) --acpp-cuda-path=$(CUDA_PATH) -o $@ $^

run_sycl_test_acpp_cuda: sycl_test_acpp
	ACPP_VISIBILITY_MASK=cuda CUDA_VISIBILE_DEVICES=0 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib ./sycl_test_acpp

debug_sycl_test_acpp_cuda: sycl_test_acpp
	ACPP_VISIBILITY_MASK=cuda CUDA_VISIBILE_DEVICES=0 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib gdb ./sycl_test_acpp

run_sycl_test_acpp_hip: sycl_test_acpp
	ACPP_VISIBILITY_MASK=hip ./sycl_test_acpp

debug_sycl_test_acpp_hip: sycl_test_acpp
	ACPP_VISIBILITY_MASK=hip rocgdb ./sycl_test_acpp
	#ACPP_VISIBILITY_MASK=hip HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 rocgdb ./sycl_test_acpp

run_sycl_test_acpp_omp: sycl_test_acpp
	echo "USING OMP from /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib..."
	#LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib ./sycl_test_acpp
	ACPP_VISIBILITY_MASK=omp LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib ./sycl_test_acpp

debug_sycl_test_acpp_omp: sycl_test_acpp
	ACPP_VISIBILITY_MASK=omp LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/lib gdb ./sycl_test_acpp



run_sycl_test_dpcpp_hip: sycl_test_dpcpp_hip
	echo $(ONEAPI_LIB_DEP)
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./sycl-implementations/${HOST}/dpc++-hip/lib ./$^

run_sycl_test_dpcpp_cuda: sycl_test_dpcpp_cuda
	echo $(ONEAPI_LIB_DEP)
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./sycl-implementations/${HOST}/dpc++-cuda/lib:${CUDA_PATH}/extras/CUPTI/lib64 ./$^

clean:
	rm -f sycl_wrapper.*.so\
		libsampler-*.so\
		sycl_test_*
	rm -fr acpp dpcp usy
