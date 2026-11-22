#!/bin/bash
# Unified fine-tuning script for ManipLLM.
#
# Usage:
#   bash finetune.sh [DISTANCE] [GPU_ID]
#
#   DISTANCE  random | FLOAT   camera distance used during data collection
#               random  - randomly-placed camera (original ManipLLM baseline)
#               FLOAT   - fixed distance, e.g. 2.0, 3.0, 4.0, 5.0
#             (default: random)
#   GPU_ID    GPU index for CUDA_VISIBLE_DEVICES  (default: 0)
#
# Examples:
#   bash finetune.sh                  # train on random-distance data, GPU 0
#   bash finetune.sh 3.0              # train on 3.0m fixed-distance data, GPU 0
#   bash finetune.sh 4.0 1            # train on 4.0m data, GPU 1

set -e

DISTANCE="${1:-random}"
GPU_ID="${2:-0}"

# ── Resolve directories and port from distance ────────────────────────────
if [ "$DISTANCE" = "random" ]; then
    DATA_DIR="../data_collection/data/train_data_random"
    JSON_DIR="./data/train_json_random"
    OUTPUT_DIR="./exp/train_ckpts"
    MASTER_PORT=11710
else
    DATA_DIR="../data_collection/data/train_data_dist${DISTANCE}"
    JSON_DIR="./data/train_json_dist${DISTANCE}"
    OUTPUT_DIR="./exp/train_ckpts_dist${DISTANCE}"
    # Use a stable port derived from the distance to avoid conflicts when
    # running multiple jobs in parallel (e.g. 3.0→11730, 4.0→11740).
    PORT_OFFSET=$(python3 -c "print(int(float('${DISTANCE}') * 10))")
    MASTER_PORT=$((11700 + PORT_OFFSET))
fi

echo "=================================================="
echo "  ManipLLM Fine-tuning"
echo "=================================================="
echo "  Distance:    ${DISTANCE}"
echo "  Data dir:    ${DATA_DIR}"
echo "  JSON dir:    ${JSON_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  GPU:         ${GPU_ID}  (master port: ${MASTER_PORT})"
echo "=================================================="
echo ""

# Step 1 — build training JSON
echo "[1/2] Generating training JSON..."
python ./data/create_dataset_aff.py \
    --folder_dir "${DATA_DIR}" \
    --output_dir "${JSON_DIR}" \
    --num_point  20

# Step 2 — fine-tune
echo "[2/2] Starting fine-tuning..."
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u \
    -m torch.distributed.launch \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=1 \
    --use_env \
    main_finetune.py \
    --batch_size    4 \
    --epochs        10 \
    --warmup_epochs 1 \
    --blr           1e-3 \
    --weight_decay  0.02 \
    --output_dir    "${OUTPUT_DIR}" \
    --pretrained_path ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth \
    --llama_path      ./ckpts/llama_model_weights \
    --bins  True \
    --mlm   True \
    --aff_prior \
    --data_config  "${JSON_DIR}"

echo ""
echo "Training complete. Checkpoints saved to: ${OUTPUT_DIR}"
