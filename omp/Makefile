include ./makefile_defs.mk

WRAPPER_NAME:=omp_wrapper$(shell python3-config --extension-suffix)
all: $(WRAPPER_NAME)

###############################################################################
#                                  Sampler                                    #
###############################################################################
omp_libsampler.so: omp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(CXX) $(CXXFLAGS) -O3 -DWRAPPER_BUILD=1 -std=c++17 -march=native -ffast-math -shared -fPIC -fopenmp -lgomp -o $@ $^
	#$(CXX) $(CXXFLAGS) -g -O0 -DWRAPPER_BUILD=1 -std=c++17 -march=native -ffast-math -shared -fPIC -fopenmp -lgomp -o $@ $^

$(WRAPPER_NAME): omp_libsampler.so pybind_wrapper.cpp
	$(CXX) $(CXXFLAGS) -O3 -Wall -Werror -DWRAPPER_BUILD=1 -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. $^ \
		-ltorch -fopenmp \
		-march=native \
		omp_libsampler.so \
		-o $@\
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

omp_test: omp_test.cpp omp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(CXX) -g -O0 -std=c++17 $(CXXFLAGS) -fopenmp -o $@ $^

omp_instrument: omp_test.cpp omp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(CXX) -DINSTRUMENTATION -g -O3 -std=c++17 $(CXXFLAGS) -fopenmp -o $@ $^

omp_test_flto: omp_test.cpp omp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(CXX) -g -O3 -std=c++17 $(CXXFLAGS) -fopenmp -flto -o $@ $^

omp_instrument_flto: omp_test.cpp omp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp ../legacy/cpp_sampler.cpp
	$(CXX) -DINSTRUMENTATION -g -O3 -std=c++17 $(CXXFLAGS) -fopenmp -flto -o $@ $^

clean:
	rm -f $(WRAPPER_NAME) omp_libsampler.so\
		*.csv\
		omp_test\
		omp_test_flto

