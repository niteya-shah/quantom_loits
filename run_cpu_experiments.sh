#!/usr/bin/env bash
set -euo pipefail

# Run from inside an allocated Instinct compute node/job.
# By default all physical CPU cores visible to the allocation are used.
# Override the maximum explicitly with, for example:
#   NTHREADS=64 ./run_cpu_experiments.sh

SITE="${SITE:-instinct}"
GRID="${GRID:-100}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
SEED="${SEED:-0}"

STRONG_EVENTS="${STRONG_EVENTS:-100000}"
EVENTS_PER_THREAD="${EVENTS_PER_THREAD:-10000}"
read -r -a FIXED_EVENTS <<< "${FIXED_EVENTS:-10000 100000 1000000 10000000}"

ACPP_VARIANT="${ACPP_VARIANT:-acpp-instinct-cpu}"
DPCPP_VARIANT="${DPCPP_VARIANT:-dpcpp-instinct-cpu}"

OUTPUT="${OUTPUT:-results/training/${SITE}-cpu}"
PLOTS="${PLOTS:-results/plots/${SITE}-cpu}"

command -v taskset >/dev/null 2>&1 || {
    echo "ERROR: taskset is required for CPU scaling affinity." >&2
    exit 2
}
command -v lscpu >/dev/null 2>&1 || {
    echo "ERROR: lscpu is required to identify physical CPU cores." >&2
    exit 2
}

# Select one logical CPU from each physical core, restricted to the CPU affinity
# already granted to this job. This avoids accidentally scaling over SMT threads.
mapfile -t PHYSICAL_CPUS < <(
    python - <<'PY'
import os
import subprocess

allowed = os.sched_getaffinity(0)
text = subprocess.check_output(
    ["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True
)
seen = set()
for line in text.splitlines():
    if not line or line.startswith("#"):
        continue
    cpu_s, core_s, socket_s, online = line.split(",")
    cpu = int(cpu_s)
    if cpu not in allowed or online != "Y":
        continue
    physical = (int(socket_s), int(core_s))
    if physical in seen:
        continue
    seen.add(physical)
    print(cpu)
PY
)

AVAILABLE_CORES="${#PHYSICAL_CPUS[@]}"
if (( AVAILABLE_CORES == 0 )); then
    echo "ERROR: no physical CPU cores are visible to this process." >&2
    exit 2
fi

NTHREADS="${NTHREADS:-$AVAILABLE_CORES}"
if ! [[ "$NTHREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NTHREADS must be a positive integer." >&2
    exit 2
fi
if (( NTHREADS > AVAILABLE_CORES )); then
    echo "ERROR: requested NTHREADS=$NTHREADS but only $AVAILABLE_CORES physical cores are visible." >&2
    exit 2
fi

THREADS=()
t=1
while (( t <= NTHREADS )); do
    THREADS+=("$t")
    t=$((t * 2))
done
if (( THREADS[${#THREADS[@]} - 1] != NTHREADS )); then
    THREADS+=("$NTHREADS")
fi

cpu_list_for_threads() {
    local threads="$1"
    local selected=("${PHYSICAL_CPUS[@]:0:threads}")
    local IFS=,
    echo "${selected[*]}"
}

for variant in "$ACPP_VARIANT" "$DPCPP_VARIANT"; do
    if [[ ! -f "sycl/build/$variant/variant.py" || ! -f "sycl/build/$variant/libquantom_loits_sycl.so" ]]; then
        echo "ERROR: SYCL variant '$variant' is not built." >&2
        echo "Build the Instinct CPU toolchains/backends before running this script." >&2
        exit 2
    fi
done

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=FALSE
export QUANTOM_SITE="$SITE"

mkdir -p "$OUTPUT" "$PLOTS"

echo "============================================================"
echo "Instinct CPU scaling study"
echo "Physical cores visible: $AVAILABLE_CORES"
echo "Maximum cores used:      $NTHREADS"
echo "Scaling points:          ${THREADS[*]}"
echo "Strong workload:         $STRONG_EVENTS events"
echo "Weak workload:           $EVENTS_PER_THREAD events/core"
echo "AdaptiveCpp:             $ACPP_VARIANT"
echo "DPC++:                   $DPCPP_VARIANT"
echo "============================================================"

echo
echo "Building C++ and OpenMP backends"
make build-cpp
make build-openmp

run_benchmark() {
    local backend="$1"
    local events="$2"
    local threads="$3"
    local variant="${4:-}"
    local cpus
    cpus="$(cpu_list_for_threads "$threads")"

    echo
    echo "------------------------------------------------------------"
    if [[ -n "$variant" ]]; then
        echo "backend=$backend variant=$variant events=$events cores=$threads cpus=$cpus"
    else
        echo "backend=$backend events=$events cores=$threads cpus=$cpus"
    fi
    echo "------------------------------------------------------------"

    local env_args=(
        "OMP_NUM_THREADS=$threads"
        "MKL_NUM_THREADS=$threads"
        "OPENBLAS_NUM_THREADS=$threads"
        "NUMEXPR_NUM_THREADS=$threads"
        "QUANTOM_CPU_THREADS=$threads"
        "QUANTOM_SITE=$SITE"
    )

    if [[ "$variant" == "$ACPP_VARIANT" ]]; then
        env_args+=(
            "QUANTOM_SYCL_VARIANT=$variant"
            "ACPP_VISIBILITY_MASK=omp"
        )
    elif [[ "$variant" == "$DPCPP_VARIANT" ]]; then
        env_args+=(
            "QUANTOM_SYCL_VARIANT=$variant"
            "ONEAPI_DEVICE_SELECTOR=native_cpu:cpu"
            "SYCL_NATIVE_CPU_HOST_THREADS=$threads"
        )
    fi

    taskset -c "$cpus" \
        env "${env_args[@]}" \
        python benchmark_training.py \
            --backend "$backend" \
            --device cpu \
            --events "$events" \
            --grid-size "$GRID" \
            --warmup "$WARMUP" \
            --iterations "$ITERS" \
            --seed "$SEED" \
            --site "$SITE" \
            --output "$OUTPUT" \
            --regions
}

run_parallel_backends() {
    local events="$1"
    local threads="$2"

    run_benchmark openmp "$events" "$threads"
    run_benchmark torch "$events" "$threads"
    run_benchmark sycl "$events" "$threads" "$ACPP_VARIANT"
    run_benchmark sycl "$events" "$threads" "$DPCPP_VARIANT"
}

echo
echo "============================================================"
echo "1. Fixed-resource scaling"
echo "============================================================"

for events in "${FIXED_EVENTS[@]}"; do
    run_benchmark cpp "$events" "$NTHREADS"
    run_parallel_backends "$events" "$NTHREADS"
done

echo
echo "============================================================"
echo "2. Strong scaling"
echo "   Fixed workload: $STRONG_EVENTS events"
echo "============================================================"

for threads in "${THREADS[@]}"; do
    run_parallel_backends "$STRONG_EVENTS" "$threads"
done

echo
echo "============================================================"
echo "3. Weak scaling"
echo "   $EVENTS_PER_THREAD events/core"
echo "============================================================"

for threads in "${THREADS[@]}"; do
    events=$((threads * EVENTS_PER_THREAD))
    run_parallel_backends "$events" "$threads"
done

echo
echo "============================================================"
echo "4. Generate plots"
echo "============================================================"

python - <<PY
from plotting.plot_ss import generate
assert generate("$OUTPUT", "$PLOTS/strong_scaling.pdf")
PY

python - <<PY
from plotting.plot_ws import generate
assert generate("$OUTPUT", "$PLOTS/weak_scaling.pdf")
PY

python -m plotting.plot_fixed_resource \
    --input "$OUTPUT" \
    --output "$PLOTS/cpu_scaling.pdf"

echo
echo "Done."
echo "Results: $OUTPUT/"
echo "Plots:   $PLOTS/"
