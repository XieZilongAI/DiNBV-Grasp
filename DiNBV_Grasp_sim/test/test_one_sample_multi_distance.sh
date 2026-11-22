#!/bin/bash
# Test one sample with multiple camera distances
# Example: test 103866_Printer_5_pulling_14 with all 60 distances

RECORD_NAME="100720_Globe_37_pulling_4"
DATA_DIR="../data_collection/data/test_data_camera_distances"
OUT_DIR="./test_results/distance_eval_results"
USE_MASK="False"

echo "Testing sample: $RECORD_NAME"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUT_DIR"
echo ""

python test_multi_distance_in_sapien.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --record_name "$RECORD_NAME" \
  --use_mask "$USE_MASK" \
  --no_gui \
  --num_processes 4

echo ""
echo "Done! Results saved to $OUT_DIR/$RECORD_NAME/dist_*.json"

