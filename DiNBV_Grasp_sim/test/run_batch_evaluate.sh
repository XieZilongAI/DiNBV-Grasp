#!/bin/bash
# Batch evaluate all camera distances to generate labels

LLAMA_DIR="/data1/xzl/ManipLLM/train/ckpts/llama_model_weights"
ADAPTER_DIR="/data1/xzl/ManipLLM/train/exp/train_ckpts/checkpoint-9.pth"  # path to your best checkpoint
DATA_DIR="/data1/xzl/ManipLLM/data_collection/data/test_data_camera_distances_v2"
OUTPUT_DIR="/data1/xzl/ManipLLM/test/test_results/distance_eval_results_test_v2"

# Set GPUs to use, e.g. "0,1,2,3" or "0,1".
# If not set, the script will use all available GPUs.
# GPU_IDS="${GPU_IDS:-0,1,2}"  # default GPUs, can be overridden via env

echo "Starting batch evaluation of camera distances..."
echo "This will generate labels for ~96,500 images"
echo "Using GPUs: $GPU_IDS"

# Step 1: Model inference with multi-GPU support
CUDA_VISIBLE_DEVICES=$GPU_IDS python batch_evaluate_distances.py \
  --llama_dir "$LLAMA_DIR" \
  --adapter_dir "$ADAPTER_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --action pulling \
  --num_gpus $(echo $GPU_IDS | tr ',' '\n' | wc -l)

echo "Model inference complete!"
echo ""
echo "Next step: Evaluate predictions in SAPIEN simulator"
echo "This will take several hours..."
echo ""

# Step 2: SAPIEN simulation evaluation
python test_multi_distance_in_sapien.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUTPUT_DIR" \
  --num_processes 20 \
  --no_gui

echo "Evaluation complete! Results in $OUTPUT_DIR"



