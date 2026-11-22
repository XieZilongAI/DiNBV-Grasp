#!/usr/bin/env python3
"""
Incremental copy results from distance_eval_results to distance_eval_results_test
Only copy files that don't exist in the destination
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def incremental_copy(src_dir, dst_dir):
    """
    Copy files from src_dir to dst_dir, skipping files that already exist in dst_dir
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        return
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Get all files in source directory recursively
    all_files = list(src_path.rglob('*'))
    files_to_copy = [f for f in all_files if f.is_file()]
    
    print(f"Found {len(files_to_copy)} files in source directory")
    print(f"Copying to {dst_dir}...")
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for src_file in tqdm(files_to_copy, desc="Copying files"):
        # Get relative path from source directory
        rel_path = src_file.relative_to(src_path)
        dst_file = dst_path / rel_path
        
        # Skip if destination file already exists
        if dst_file.exists():
            skipped_count += 1
            continue
        
        # Create destination directory if needed
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        except Exception as e:
            print(f"\nError copying {src_file}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Copy completed!")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    src_dir = '/data1/xzl/ManipLLM/test/test_results/distance_eval_results'
    dst_dir = '/data1/xzl/ManipLLM/test/test_results/distance_eval_results_test_v1'
    
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print()
    
    incremental_copy(src_dir, dst_dir)

