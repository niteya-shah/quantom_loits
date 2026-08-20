#!/usr/bin/env bash

# Run from inside an allocated GPU compute node/job with the site's Python/GPU
# environment already active. The site and GPU backend are selected from the
# short hostname:
#   odyssey* / *odyssey* -> AMD MI300A
#   illyad*  / *illyad*  -> NVIDIA H100
#   x*                    -> Aurora PVC compute nodes
#
# Override the event sweep, for example:
#   FIXED_EVENTS="1000000 10000000" ./run_gpu_experiments.sh
#
# Set REGIONS=0 to collect only clean wall-clock timing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="$(hostname -s)"

case "$HOST" in
    *odyssey*)
        SITE=odyssey
        DEVICE=cuda
        GPU_NAME="AMD MI300A"
        SYCL_VARIANTS=(
            acpp-odyssey-mi300a
            dpcpp-odyssey-mi300a
        )
        ;;
    *illyad*)
        SITE=illyad
        DEVICE=cuda
        GPU_NAME="NVIDIA H100"
        SYCL_VARIANTS=(
            acpp-illyad-h100
            dpcpp-illyad-h100
        )
        ;;
    x*)
        SITE=aurora
        DEVICE=xpu
        GPU_NAME="Intel PVC"
        SYCL_VARIANTS=(
            dpcpp-aurora-pvc
        )
        # FLAT exposes one PVC tile per XPU device and avoids Level Zero
        # implicit two-tile scaling. Respect an explicit caller override.
        export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
        ;;
    *)
        echo "ERROR: unsupported GPU host '$HOST'." >&2
        echo "Expected an Odyssey host, an Illyad host, or an Aurora x* compute node." >&2
        exit 2
        ;;
esac

GRID="${GRID:-100}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-25}"
SEED="${SEED:-0}"
REGIONS="${REGIONS:-1}"
read -r -a FIXED_EVENTS <<< "${FIXED_EVENTS:-100000 1000000 10000000 100000000 1000000000}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/training}"
SITE_PLOTS="${SITE_PLOTS:-results/plots/${SITE}}"
GPU_PLOTS="${GPU_PLOTS:-results/plots/gpu}"

verify_sycl_variant() {
    local variant="$1"

    if [[ ! -f "sycl/build/$variant/variant.py" || ! -f "sycl/build/$variant/libquantom_loits_sycl.so" ]]; then
        echo "ERROR: SYCL variant '$variant' is not built." >&2
        echo "Build the required GPU SYCL backend before running this script." >&2
        exit 2
    fi

    if ! python - "$variant" <<'PY'
from pathlib import Path
import sys

variant = sys.argv[1]
expected = (64, 2, 64, 2)
namespace = {}
exec((Path("sycl/build") / variant / "variant.py").read_text(), namespace)
metadata = namespace["METADATA"]
keys = (
    "vjp_team_size",
    "vjp_items_per_lane",
    "compact_team_size",
    "compact_items_per_lane",
)
actual = tuple(int(metadata.get(key, -1)) for key in keys)
if actual != expected:
    raise SystemExit(
        f"{variant}: stale/non-final launch geometry {actual}; expected {expected}"
    )
PY
    then
        echo "ERROR: rebuild '$variant' with the finalized SYCL build scripts." >&2
        exit 2
    fi
}

for variant in "${SYCL_VARIANTS[@]}"; do
    verify_sycl_variant "$variant"
done

mkdir -p "$OUTPUT_ROOT" "$SITE_PLOTS" "$GPU_PLOTS"

echo "============================================================"
echo "$SITE GPU fixed-resource study"
echo "Host:             $HOST"
echo "GPU:              $GPU_NAME"
echo "Torch device:     $DEVICE"
echo "Event counts:     ${FIXED_EVENTS[*]}"
echo "Grid:             $GRID"
echo "Warmups:          $WARMUP"
echo "Iterations:       $ITERS"
echo "Seed:             $SEED"
echo "Region profiling: $REGIONS"
if [[ "$SITE" == "aurora" ]]; then
    echo "ZE hierarchy:     $ZE_FLAT_DEVICE_HIERARCHY"
fi
echo "SYCL variants:    ${SYCL_VARIANTS[*]}"
echo "============================================================"

run_benchmark() {
    local backend="$1"
    local events="$2"
    local variant="${3:-}"
    local region_args=()

    if [[ "$REGIONS" != "0" ]]; then
        region_args+=(--regions)
    fi

    echo
    echo "------------------------------------------------------------"
    if [[ -n "$variant" ]]; then
        echo "backend=$backend variant=$variant events=$events"
    else
        echo "backend=$backend events=$events"
    fi
    echo "------------------------------------------------------------"

    (
        export QUANTOM_SITE="$SITE"
        if [[ -n "$variant" ]]; then
            export QUANTOM_SYCL_VARIANT="$variant"
        else
            unset QUANTOM_SYCL_VARIANT || true
        fi

        python benchmark_training.py \
            --backend "$backend" \
            --device "$DEVICE" \
            --events "$events" \
            --grid-size "$GRID" \
            --warmup "$WARMUP" \
            --iterations "$ITERS" \
            --seed "$SEED" \
            --site "$SITE" \
            --experiment fixed \
            --output "$OUTPUT_ROOT" \
            "${region_args[@]}"
    )
}

for events in "${FIXED_EVENTS[@]}"; do
    run_benchmark torch "$events"
    for variant in "${SYCL_VARIANTS[@]}"; do
        run_benchmark sycl "$events" "$variant"
    done
done

if [[ "$REGIONS" != "0" ]]; then
    echo
    echo "============================================================"
    echo "Generate GPU fixed-resource plots"
    echo "============================================================"

    python - <<PY
from plotting.plot_fixed_resource import generate

assert generate(
    ["$OUTPUT_ROOT"],
    "$SITE_PLOTS/fixed.pdf",
    cpu=False,
    site="$SITE",
)

# The filesystem is shared across sites, so refresh the aggregate GPU plot from
# whatever complete GPU datasets are currently present.
generate(
    ["$OUTPUT_ROOT"],
    "$GPU_PLOTS/fixed.pdf",
    cpu=False,
)
PY
fi

echo
echo "Done."
echo "Results: $OUTPUT_ROOT/$SITE/fixed/"
if [[ "$REGIONS" != "0" ]]; then
    echo "Plot:    $SITE_PLOTS/fixed.pdf"
    echo "GPU plot:$GPU_PLOTS/fixed.pdf"
fi
