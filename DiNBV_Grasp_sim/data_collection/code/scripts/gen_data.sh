#!/bin/bash
# Unified data generation script for ManipLLM.
#
# Usage:
#   bash gen_data.sh [OPTIONS]
#
# Options:
#   --mode       train | test           (default: test)
#   --distance   random | multi | FLOAT (default: multi)
#                  random  - random camera placement (backbone training data)
#                  multi   - sweep 1.0m-6.0m every 0.1m (distance-selector data)
#                  FLOAT   - fixed distance, e.g. 2.0, 3.0, 4.0, 5.0
#   --processes  N                      (default: 10)
#   --epochs     N                      (default: 20)
#   --start      N  starting epoch      (default: 0)
#
# Examples:
#   # Collect backbone training data (~20,000 samples, random distance)
#   bash gen_data.sh --mode train --distance random --processes 24 --epochs 100 --start 50
#
#   # Collect multi-distance test data for the distance selector (~1,500 samples)
#   bash gen_data.sh --mode test --distance multi --processes 10 --epochs 20 --start 10
#
#   # Collect fixed-distance data at 3.0 m
#   bash gen_data.sh --mode train --distance 3.0 --processes 16 --epochs 50

set -e

# ── Defaults ────────────────────────────────────────────────────────────────
MODE="test"
DISTANCE="multi"
NUM_PROCESSES=10
NUM_EPOCHS=20
STARTING_EPOCH=0
# ────────────────────────────────────────────────────────────────────────────

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";          shift 2 ;;
        --distance)   DISTANCE="$2";      shift 2 ;;
        --processes)  NUM_PROCESSES="$2"; shift 2 ;;
        --epochs)     NUM_EPOCHS="$2";    shift 2 ;;
        --start)      STARTING_EPOCH="$2";shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve paths and flags based on mode/distance ──────────────────────────
EXTRA_FLAGS=""

if [ "$DISTANCE" = "random" ]; then
    DATA_DIR="../data/${MODE}_data_random"
    DESC="random camera distance"
elif [ "$DISTANCE" = "multi" ]; then
    DATA_DIR="../data/${MODE}_data_camera_distances_v2"
    EXTRA_FLAGS="--save_approach_trajectory"
    DESC="multi-distance sweep (1.0m–6.0m, step 0.1m)"
else
    DATA_DIR="../data/${MODE}_data_dist${DISTANCE}"
    EXTRA_FLAGS="--camera_distance ${DISTANCE}"
    DESC="fixed distance ${DISTANCE}m"
fi

# Use the unified generator; legacy *_v1 variants have been removed.
GEN_SCRIPT="gen_offline_data.py"

DATA_FN="../stats/${MODE}_id.txt"
INS_CNT_FN="../stats/ins_cnt_46cats.txt"

echo "=================================================="
echo "  ManipLLM Data Generation"
echo "=================================================="
echo "  Mode:        ${MODE}"
echo "  Distance:    ${DESC}"
echo "  Output dir:  ${DATA_DIR}"
echo "  Processes:   ${NUM_PROCESSES}"
echo "  Epochs:      ${NUM_EPOCHS}  (start: ${STARTING_EPOCH})"
echo "=================================================="
echo ""

python ${GEN_SCRIPT} \
    --data_dir      "${DATA_DIR}" \
    --data_fn       "${DATA_FN}" \
    --primact_types pulling \
    --num_processes "${NUM_PROCESSES}" \
    --num_epochs    "${NUM_EPOCHS}" \
    --starting_epoch "${STARTING_EPOCH}" \
    --ins_cnt_fn    "${INS_CNT_FN}" \
    --mode          "${MODE}" \
    ${EXTRA_FLAGS}

echo ""
echo "Done. Data saved to: ${DATA_DIR}"
