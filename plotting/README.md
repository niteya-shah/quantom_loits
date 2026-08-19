# Plotting

The plotting tools use clustered/stacked bar plots, hatching for event counts,
and log-scaled execution-time panels where appropriate. Timing bars use mean ±
one sample standard deviation.

Benchmark results use one canonical tree:

    results/training/<site>/<experiment>/
        training_<implementation>_<device>_e<events>[_t<threads>].csv
        regions_<implementation>_<device>_e<events>[_t<threads>].csv
        trace_<implementation>_<device>_e<events>[_t<threads>].json

where `<experiment>` is `fixed`, `strong`, `weak`, or `tuning`. SYCL tuning
runs add one directory level, for example `tuning/vjp4-compact2/`, so different
compile-time tuning cases cannot overwrite one another.

Every CSV row records the site, experiment, backend, implementation, device,
event count, thread count, grid size, warmup count, measured iteration count,
seed, and the selected SYCL VJP/compaction cases. Plotting uses those columns as
the source of truth rather than recovering metadata from filenames.

`training_*.csv` contains synchronized wall-clock GAN-iteration timings and is
not collected under `torch.profiler`. `regions_*.csv` is collected separately
with the profiler and is used only for the LOITS component breakdown.

The fixed-resource top panel is the percentage of LOITS forward+backward time in
each semantic stage. The lower panel is unprofiled end-to-end GAN iteration
time. Strong and weak scaling also use the unprofiled wall-clock timings.

Generate all available plots with:

    make plots

Outputs are organized as:

    results/plots/gpu/fixed.pdf
    results/plots/<cpu-site>/fixed.pdf
    results/plots/<cpu-site>/strong.pdf
    results/plots/<cpu-site>/weak.pdf
