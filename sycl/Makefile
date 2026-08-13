.PHONY: all acpp acpp-cpu acpp-cuda acpp-hip dpcpp-cpu dpcpp-xpu dpcpp-cuda dpcpp-hip \
        dpc++_for_tbb dpc++_for_cuda dpc++_for_hip dpc++_for_spirv test clean

all: acpp

acpp:
	./build-acpp.sh generic

acpp-cpu:
	./build-acpp.sh cpu

acpp-cuda:
	./build-acpp.sh cuda

acpp-hip:
	./build-acpp.sh hip

dpcpp-cpu:
	./build-dpcpp.sh cpu

dpcpp-xpu:
	./build-dpcpp.sh xpu

dpcpp-cuda:
	./build-dpcpp.sh cuda

dpcpp-hip:
	./build-dpcpp.sh hip

# Temporary aliases for the old top-level Makefile until the repository cleanup patch.
dpc++_for_tbb: dpcpp-cpu
dpc++_for_cuda: dpcpp-cuda
dpc++_for_hip: dpcpp-hip
dpc++_for_spirv: dpcpp-xpu

test:
	python -m pytest -q ../tests/test_sycl_loits.py

clean:
	rm -rf build
