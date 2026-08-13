# Plotting

The plotting tools intentionally use the same presentation style as the original
QuantOm artifact: clustered/stacked **bar plots**, hatching for event counts,
and log-scaled execution-time panels where appropriate. There are no boxplots or
median/IQR plots.

All timing bars use **mean ± one sample standard deviation** error bars.

The new differentiable benchmark is represented directly in the old style:

- the top CPU/GPU panel is the percentage of LOITS runtime spent in each
  component;
- forward stages are labelled `F:` and reverse-mode stages are labelled `B:`;
- the two forward sampler calls and one backward call in each GAN iteration are
  accumulated together before the component percentages are computed;
- the bottom CPU/GPU panel is the unprofiled end-to-end GAN iteration time with
  standard-deviation error bars.

The component plots use semantic leaf regions only, so nested profiler parents
are never double-counted. PyTorch-only flatten regions are included when they
exist; native backends simply contribute zero to those components.

## Generate plots

The normal interface is just:

    make plots

or:

    python -m plotting.plot_all

This reads `results/training/` and writes, when enough data exists:

    results/plots/cpu_scaling.pdf
    results/plots/gpu_scaling.pdf
    results/plots/strong_scaling.pdf
    results/plots/weak_scaling.pdf

CPU/GPU plots require the normal benchmark CSV and the detailed `--regions`
CSV for the same configurations. `benchmark_training.py --regions` already
produces both.

Strong scaling is generated automatically when multiple explicit OpenMP thread
counts exist for the same event count and a serial C++ result exists. Weak
scaling is generated when the results contain at least two thread counts with a
constant events-per-thread workload. Missing plots are skipped by `plot_all`.

No backend is required. The scripts plot only the result series present in the
shared results directory, including explicitly named SYCL variants.
