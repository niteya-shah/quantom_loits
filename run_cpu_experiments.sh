#!/usr/bin/env bash
set -euo pipefail

# Run from inside an allocated compute node/job.
#
# Required:
#   NTHREADS=<number of physical CPU cores allocated> ./run_cpu_experiments.sh
#
# Example:
#   NTHREADS=64 ./run_cpu_experiments.sh

if [[ -z "${NTHREADS:-}" ]]; then
    echo "ERROR: NTHREADS must be set explicitly to the number of physical CPU cores allocated."
    echo "Example: NTHREADS=64 ./run_cpu_experiments.sh"
    exit 2
fi

if ! [[ "${NTHREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NTHREADS must be a positive integer."
    exit 2
fi

GRID=100
WARMUP=5
ITERS=10
SEED=0

FIXED_EVENTS=(10000 100000 1000000)
STRONG_EVENTS=1000000
EVENTS_PER_THREAD=10000

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE

mkdir -p results/training results/plots

echo "============================================================"
echo "Building C++ and OpenMP backends"
echo "============================================================"
make build-cpp
make build-openmp

run_benchmark() {
    local backend="$1"
    local events="$2"
    local threads="$3"

    echo
    echo "------------------------------------------------------------"
    echo "backend=${backend} events=${events} threads=${threads}"
    echo "------------------------------------------------------------"

    OMP_NUM_THREADS="${threads}" \
    MKL_NUM_THREADS="${threads}" \
    python benchmark_training.py \
        --backend "${backend}" \
        --device cpu \
        --events "${events}" \
        --grid-size "${GRID}" \
        --warmup "${WARMUP}" \
        --iterations "${ITERS}" \
        --seed "${SEED}" \
        --regions
}

echo
echo "============================================================"
echo "1. Fixed-resource C++ vs OpenMP"
echo "============================================================"

for events in "${FIXED_EVENTS[@]}"; do
    run_benchmark cpp "${events}" "${NTHREADS}"
    run_benchmark openmp "${events}" "${NTHREADS}"
done

echo
echo "============================================================"
echo "2. OpenMP strong scaling"
echo "   Fixed workload: ${STRONG_EVENTS} events"
echo "============================================================"

t=1
while (( t <= NTHREADS )); do
    run_benchmark openmp "${STRONG_EVENTS}" "${t}"
    t=$((t * 2))
done

# Include the full allocation even when NTHREADS is not a power of two.
last_power=$((t / 2))
if (( last_power != NTHREADS )); then
    run_benchmark openmp "${STRONG_EVENTS}" "${NTHREADS}"
fi

echo
echo "============================================================"
echo "3. OpenMP weak scaling"
echo "   ${EVENTS_PER_THREAD} events/thread"
echo "============================================================"

t=1
while (( t <= NTHREADS )); do
    events=$((t * EVENTS_PER_THREAD))
    run_benchmark openmp "${events}" "${t}"
    t=$((t * 2))
done

# Include the full allocation even when NTHREADS is not a power of two.
last_power=$((t / 2))
if (( last_power != NTHREADS )); then
    events=$((NTHREADS * EVENTS_PER_THREAD))
    run_benchmark openmp "${events}" "${NTHREADS}"
fi

echo
echo "============================================================"
echo "4. Generate plots"
echo "============================================================"
make plots

echo
echo "Done."
echo "Results: results/training/"
echo "Plots:   results/plots/"
