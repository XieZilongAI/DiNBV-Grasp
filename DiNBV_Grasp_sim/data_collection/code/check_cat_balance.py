import os
import numpy as np
import random
import shutil

data_dir = '/data1/xzl/ManipLLM/data_collection/data/test_data_camera_distances'
data_list = os.listdir(data_dir)
cat_cal = dict()
cat_files = dict()  # store the file list for each category

# Count how many samples each category has
for data_name in data_list:
    cat = data_name.split('_')[1]
    if cat not in list(cat_cal.keys()):
        cat_cal[cat] = 1
        cat_files[cat] = [data_name]
    else:
        cat_cal[cat] += 1
        cat_files[cat].append(data_name)

print("Original statistics:", cat_cal, len(cat_cal))

# Balance data: randomly keep 50 samples per category
target_count = 50
deleted_count = 0
for cat, files in cat_files.items():
    if len(files) > target_count:
        # Randomly choose 50 files to keep
        random.shuffle(files)
        files_to_keep = files[:target_count]
        files_to_delete = files[target_count:]
        
        # Delete extra files/directories
        for file_to_delete in files_to_delete:
            file_path = os.path.join(data_dir, file_to_delete)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                deleted_count += 1
        
        print(f"{cat}: keep {len(files_to_keep)}, delete {len(files_to_delete)}")
    else:
        print(f"{cat}: keep {len(files)} (< {target_count}, keep all)")

print(f"\nTotal deleted files: {deleted_count}")

# Re-count balanced data
data_list_after = os.listdir(data_dir)
cat_cal_after = dict()
for data_name in data_list_after:
    cat = data_name.split('_')[1]
    if cat not in list(cat_cal_after.keys()):
        cat_cal_after[cat] = 1
    else:
        cat_cal_after[cat] += 1

print("\nBalanced statistics:", cat_cal_after)