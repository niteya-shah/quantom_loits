# OpenMP LOITS backend

The OpenMP backend mirrors the C++ backend API while parallelizing the native
forward and reverse-mode VJP kernels with OpenMP. The computational core uses
raw contiguous pointers and has no Torch dependency; Torch is confined to the
binding and autograd integration layer.

Build from the repository root:

    make build-openmp

Run correctness tests:

    make test-openmp

Run the training benchmark:

    OMP_NUM_THREADS=32 python benchmark_training.py --backend openmp --device cpu --events 100000

Run the deep profiler:

    OMP_NUM_THREADS=32 python benchmark_training.py --backend openmp --device cpu --events 100000 --regions --trace
