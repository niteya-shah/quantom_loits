# QuantOm: Performance vs Productivity in HPC

This repository contains the artifact accompanying the paper:

**"On the Efficacy of PyTorch for High-Performance Computing: A Case Study in Computational Physics"**

It provides:
- implementations of the LOITS algorithm in PyTorch, C++, OpenMP, and SYCL
- scripts to reproduce all figures from the paper
- infrastructure to rerun experiments on new systems

---

## Differentiable training benchmark

The active PyTorch implementation now lives in `pytorch/`. LOITS is selected through a common backend interface:

```python
from loits import LOITS

sampler = LOITS(backend="torch", device="cuda")
events = sampler(theory_outputs, n_events)
```

The same interface is active for `cpp`; `openmp` and `sycl` will follow the same contract.

PyTorch training uses default `torch.compile(...)`. Backward compilation is provided by AOTAutograd. The stream-compaction behavior of the original LOITS implementation is preserved.

Run the end-to-end GAN benchmark with:

```bash
python benchmark_training.py --backend torch --device cuda --events 100000
python benchmark_training.py --backend cpp --device cpu --events 100000
```

Add semantic LOITS region profiling and a Chrome trace with:

```bash
python benchmark_training.py --backend torch --device cuda --events 100000 --regions --trace
python benchmark_training.py --backend cpp --device cpu --events 100000 --regions --trace
```

The C++ backend is a CPU-only PyTorch C++ extension. It consumes PyTorch tensors directly, preserves LOITS stream compaction, and implements a hand-written reverse-mode VJP for the continuous `xsec -> rho -> CDF -> interpolation -> events` path. Random samples, allocation counts, interval selection, acceptance, and compaction decisions are fixed during the reverse pass. Forward intermediates required by the VJP are saved rather than recomputed.

C++ profiling uses PyTorch C++ user scopes, so the same `torch.profiler` session captures native `rho`, `CDF`, interpolation, stream-compaction, and backward-VJP regions alongside the GAN-level ranges.

`training_*.csv` contains profiler-derived end-to-end discriminator, generator, and full training-iteration timings. `regions_*.csv` contains forward/backward LOITS region timings collected through PyTorch hooks or native C++ profiler scopes. Warmup iterations occur before profiling so compilation is excluded from steady-state measurements.

Check graph capture and the AOTAutograd path with:

```bash
python -m pytorch.compile_check
```

# 🚀 Quick Start (Recommended)

To reproduce all figures from the paper:

```bash
make
```

This will:
- copy `archived-results/` → `results/` (if needed)
- regenerate all figures into `images/`

⚠️ This does **NOT** rerun experiments. It uses archived data shipped with the artifact.

---

# ⚙️ Environment Setup (pixi)

This project uses **pixi** for environment management.

## Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Create and activate environment

```bash
pixi install
pixi shell
```

This will:
- create the environment defined in `pixi.toml`
- install all required dependencies
- ensure consistent versions across systems

---

# Reproducing Figures

Generated figures are written to:

```
images/
```

These correspond to:

- `strong_scaling.png`
- `weak_scaling.png`
- `cpu_scaling.png`
- `gpu_scaling.png`
- `stacked_barplot_cpu.pdf`
- `stacked_barplot_gpu.pdf`

---

# Running Experiments

The archived forward-only scaling data remain available for reproducing the conference figures. New differentiable measurements should use `benchmark_training.py`; the old forward-only Python benchmark has been removed.

---

# Results Layout

The artifact separates **archived paper results** from **working results**:

```
archived-results/   # results used in the paper (read-only)
results/            # working directory for regenerated or new results
```

Behavior:
- `make` → uses `results/` (bootstrapped from `archived-results/` if needed)
- `make rerun-*` → overwrites data inside `results/`

This ensures:
- paper results remain untouched
- new experiments do not clobber original data

---

# Build System

Backends are built automatically when needed. The C++ backend can also be built directly with:

```bash
make build-cpp
# or
python -m cpp.build
```

The C++ extension uses `torch.utils.cpp_extension` and writes its local build products to `cpp/build/`. The pre-existing OpenMP/SYCL code still uses `legacy/cpp_sampler.*` temporarily; the active C++ backend has no dependency on that legacy matrix/timing implementation.

---

# Backends

Available backends are detected via:

```bash
./setup-backends.sh
```

Typical configurations include:
- CPU (OpenMP, TBB)
- CUDA
- HIP
- Level Zero (XPU)

---

# Notes on DPC++ CPU (TBB)

On some systems, the DPC++ CPU backend requires the runtime library path to be visible.

This is handled automatically in the provided scripts, but if issues occur, ensure:

```
<repo>/sycl/sycl-implementations/<host>/dpc++-cpu/lib
```

is present in `LD_LIBRARY_PATH`.

---

# Notebook (Optional)

A companion Jupyter notebook (`artifact_walkthrough.ipynb`) is provided to:

- visualise generated figures
- demonstrate the workflow interactively

If using Jupyter:

```bash
pixi run python -m pip install "notebook<7"
pixi run jupyter notebook
```

(BeakerX is not required.)

---

# Cleaning

```bash
make clean
```

Removes compiled artifacts.

---

# 🧠 What This Artifact Demonstrates

This artifact supports the paper’s key findings:

- PyTorch is **4–5× more productive** (SLOC)
- CPU performance:
  - PyTorch achieves ~50–72% of optimized C++/OpenMP
- GPU performance:
  - PyTorch outperforms SYCL by:
    - ~5–6× (CUDA)
    - ~15× (HIP)
    - up to ~16× (Intel XPU)

These results arise from:
- kernel fusion
- optimized backend primitives (e.g., CUB)
- reduced synchronization overhead

---

# 📊 Experiments Included

- Strong scaling (CPU threads)
- Weak scaling (per-core workload)
- Fixed-resource scaling (CPU + GPU)
- Profiling breakdown of LOITS pipeline

---

# 📁 Repository Structure

```
.
├── pytorch/             # PyTorch LOITS, theory, GAN benchmark, profiler
├── loits.py             # uniform backend interface
├── benchmark_training.py# end-to-end differentiable benchmark
├── cpp/                 # C++ implementation
├── omp/                 # OpenMP implementation
├── sycl/                # SYCL implementations + installer
├── examples/            # Jupyter notebooks
├── archived-results/    # paper results
├── results/             # working results
├── images/              # generated figures
├── utils/               # plotting scripts
├── setup-backends.sh    # environment configuration
├── utils.sh             # logging helpers
└── Makefile
```

---

# 📓 Example Usage (Notebook)

To explore LOITS interactively:

```bash
cd examples/2d_loits
pixi run jupyter notebook
```

---

# ⚠️ Notes on Reproducibility

- GPU results depend on:
  - CUDA / ROCm / Level Zero versions
  - hardware (A100, MI300A, Intel Max)
- CPU scaling depends on:
  - NUMA configuration
  - thread pinning

We attempt to normalize this via:
- fixed seeds
- explicit thread control
- consistent build flags

---

# 🛠 Troubleshooting

### DPC++ / TBB issues
Ensure TBB is correctly built and linked:
```
sycl-implementations/<host>/tbb/
```

### CUDA / HIP not detected
Check:
```bash
echo $BACKENDS
echo $CUDA_PATH
echo $ROCM_PATH
```

### Build failures
Try:
```bash
make reinstall
```

---

# 📜 License (MIT)

Copyright 2026 Beau Johnston <beau@inbeta.org>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

# 📎 Citation

If you use this artifact, please cite:

```
Beau Johnston, Niteya Shah, Wu-chun Feng.
"On the Efficacy of PyTorch for High-Performance Computing:
A Case Study in Computational Physics."
Proceedings of the 23rd ACM International Conference on Computing Frontiers (CF 26')
10.1145/3801487.3801838
```
