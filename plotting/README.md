# Plotting

These scripts consume the current CSV schema emitted by `benchmark_training.py`.
They discover whatever result files are present; no plot requires every backend
to exist on the current machine.

Generated figures are written to `results/plots/` by default and are ignored by
Git.

## Standard plots

Generate all plots supported by the CSVs that currently exist:

    python -m plotting.plot_all --input results/training

Individual plots:

    python -m plotting.plot_training results/training
    python -m plotting.plot_loits results/training --scope autograd-forward
    python -m plotting.plot_loits results/training --scope autograd-backward
    python -m plotting.plot_speedup results/training --reference cpp
    python -m plotting.plot_breakdown results/training --events 1000000

The plotting code uses medians with interquartile ranges for scaling plots.
`plot_breakdown.py` groups the semantic LOITS regions into density, CDF, RNG,
interpolation, compaction, and reverse-VJP stages.

## SYCL results

SYCL CSVs record the selected build implementation, e.g. `acpp:cuda` or
`dpcpp:xpu`, so results from different SYCL implementations do not overwrite or
collapse into one `sycl` series. The LOITS source remains the same; this label
identifies the compiler/runtime target used for the measurement.

## Missing backends

Missing backends are normal. A machine without a built SYCL backend can still
run PyTorch/C++/OpenMP benchmarks and generate plots from those CSVs. The
plotting tools only plot series actually present in the input directory.
