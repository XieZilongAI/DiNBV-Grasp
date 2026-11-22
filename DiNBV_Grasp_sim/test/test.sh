#step1: model inference 
OUPUT_DIR='./test_results/result_dist4.0'
CUDA_VISIBLE_DEVICES=1 python test_llama.py \
  --llama_dir /data1/xzl/ManipLLM/train/ckpts/llama_model_weights \
  --adapter_dir /data1/xzl/ManipLLM/train/exp/train_ckpts_dist5.0/checkpoint-9.pth \
  --data_dir ../data_collection/data/test_data_dist5.0 \
  --out_dir "$OUPUT_DIR" \
  --action pulling

# step2: test in simulator
python test_entireprocess_in_sapien.py \
  --data_dir ../data_collection/data/test_data_dist5.0 \
  --num_processes 10 \
  --out_dir "$OUPUT_DIR" \
  --no_gui \
  --use_mask False 
  
# #step3: calculate success rate
python cal_test_mani_succ_rate.py \
    --primact_type pulling \
    --data_dir "$OUPUT_DIR"
