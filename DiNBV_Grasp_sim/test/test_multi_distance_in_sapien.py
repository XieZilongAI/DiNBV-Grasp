"""
Test multiple distance predictions in SAPIEN simulator
For each distance, load the corresponding pred_distX.X.json and test it using test_one_distance_clean.py
"""
import os
import sys
import json
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from subprocess import call
import time
import shutil


def test_one_distance(args_tuple):
    """Test one distance using test_one_distance_clean.py
    Args:
        args_tuple: (dist, data_dir, out_dir, record_name, use_mask, no_gui)
    Returns:
        (record_name, dist, ret_code) where ret_code: 0=SUCCESS, 1=exec error, 2=skip, 3=mani_succ=False, 4=already exists
    """
    dist, data_dir, out_dir, record_name, use_mask, no_gui = args_tuple
    dist_str = f"{dist:.1f}"
    pred_file = f'pred_dist{dist_str}.json'
    result_file = os.path.join(out_dir, record_name, f'dist_{dist_str}_result.json')
    
    # Check if result already exists - skip if it does
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            # Return the same code as if we just ran it
            mani_succ = result_data.get('mani_succ', False)
            if mani_succ:
                return (record_name, dist, 4)  # Already exists, SUCCESS
            else:
                return (record_name, dist, 5)  # Already exists, FAILED
        except (json.JSONDecodeError, KeyError):
            # If file exists but can't be read, re-run
            pass
    
    pred_full_path = os.path.join(out_dir, record_name, pred_file)
    if not os.path.exists(pred_full_path):
        return (record_name, dist, 2)  # Skip code - no prediction file
    
    # Run test_one_distance_clean.py with specific distance and prediction file
    cmd = f'xvfb-run -a python test_one_distance_clean.py ' \
          f'--data_dir {data_dir} ' \
          f'--record_name {record_name} ' \
          f'--out_dir {out_dir} ' \
          f'--camera_distance {dist} ' \
          f'--prediction_file {pred_file} ' \
          f'--no_gui ' \
          f'--use_mask {use_mask}'
    
    ret = call(cmd, shell=True)
    
    # Check if command executed successfully
    if ret != 0:
        if ret == 2:
            return (record_name, dist, 2)  # Skip code
        else:
            return (record_name, dist, 1)  # Failed to execute
    
    # Now check the actual result.json for mani_succ
    shape_id, category, cnt_id, primact_type, trial_id = record_name.split('_')
    # # Result is now in dist_X.X subdirectory (commented out - we use main output directory)
    # test_out_dir = os.path.join(out_dir, f'{shape_id}_{category}_{cnt_id}_{primact_type}_{int(trial_id)}', f'dist_{dist_str}')
    # result_temp = os.path.join(test_out_dir, 'result.json')
    # Result is in the main output directory (all distances write to the same directory)
    test_out_dir = os.path.join(out_dir, f'{shape_id}_{category}_{cnt_id}_{primact_type}_{int(trial_id)}')
    result_temp = os.path.join(test_out_dir, 'result.json')
    
    if not os.path.exists(result_temp):
        return (record_name, dist, 1)  # Failed
    
    # Read and check mani_succ field
    # Note: test_one_distance_clean.py now saves dist_X.X_result.json directly,
    # so we should check that file first, then fall back to result.json
    try:
        # First try to read the distance-specific result file (if it was saved by test_one_distance_clean.py)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
        elif os.path.exists(result_temp):
            # Fall back to result.json and copy it
            with open(result_temp, 'r') as f:
                result_data = json.load(f)
            # Copy result to the summary location
            shutil.copy(result_temp, result_file)
        else:
            return (record_name, dist, 1)  # Failed - no result file found
        
        # Check mani_succ field
        mani_succ = result_data.get('mani_succ', False)
        if mani_succ:
            return (record_name, dist, 0)  # SUCCESS - manipulation succeeded
        else:
            return (record_name, dist, 3)  # FAILED - manipulation failed (but execution succeeded)
    except (json.JSONDecodeError, KeyError) as e:
        return (record_name, dist, 1)  # Failed to parse result

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='original data directory (test_data_camera_distances)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory with predictions (distance_eval_results)')
    parser.add_argument('--record_name', type=str, default=None,
                        help='sample record name (if not provided, process all samples in out_dir)')
    parser.add_argument('--num_processes', type=int, default=10,
                        help='number of parallel processes')
    parser.add_argument('--no_gui', action='store_true', default=False)
    parser.add_argument('--use_mask', type=str, default='False')
    args = parser.parse_args()

    # Distances to evaluate
    distances = np.arange(1.0, 6.05, 0.1)

    # If record_name is not provided, process all samples in out_dir
    if args.record_name is None:
        # Get all sample folders from out_dir
        record_names = [f for f in os.listdir(args.out_dir) 
                        if os.path.isdir(os.path.join(args.out_dir, f))]
        print(f"Found {len(record_names)} samples in {args.out_dir}")
        print("Will process all samples...")
    else:
        record_names = [args.record_name]

    # Filter valid record names
    valid_record_names = []
    for record_name in record_names:
        sample_out_dir = os.path.join(args.out_dir, record_name)
        sample_data_dir = os.path.join(args.data_dir, record_name)
        
        if not os.path.exists(sample_out_dir):
            print(f"Warning: {sample_out_dir} does not exist, skipping {record_name}")
            continue
        
        if not os.path.exists(sample_data_dir):
            print(f"Warning: {sample_data_dir} does not exist, skipping {record_name}")
            continue
        
        valid_record_names.append(record_name)

    print(f"\n{'='*60}")
    print(f"Found {len(valid_record_names)} valid samples to test")
    print(f"Total tasks: {len(valid_record_names) * len(distances)} (samples × distances)")
    print(f"Using {args.num_processes} parallel processes")
    print(f"{'='*60}\n")

    # Generate all tasks: (dist, data_dir, out_dir, record_name, use_mask, no_gui)
    all_tasks = []
    for record_name in valid_record_names:
        for dist in distances:
            all_tasks.append((dist, args.data_dir, args.out_dir, record_name, args.use_mask, args.no_gui))

    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Process tasks in parallel using multiprocessing Pool
    print(f"Starting parallel processing with {args.num_processes} processes...")
    with mp.Pool(processes=args.num_processes) as pool:
        # Use imap for better progress tracking
        results = list(pool.imap(test_one_distance, all_tasks))
    
    # Aggregate results by record_name
    results_by_record = {}
    for record_name, dist, ret_code in results:
        if record_name not in results_by_record:
            results_by_record[record_name] = []
        results_by_record[record_name].append((dist, ret_code))
    
    # Print results for each record
    total_success = 0  # mani_succ=True
    total_fail = 0      # mani_succ=False or execution error
    total_skip = 0      # no prediction file
    total_existing = 0  # already exists (skipped execution)
    
    for record_name in valid_record_names:
        if record_name not in results_by_record:
            continue
        
        print(f"\n{'='*60}")
        print(f"Results for {record_name}")
        print(f"{'='*60}")
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        existing_count = 0
        
        # Sort by distance for consistent output
        results_by_record[record_name].sort(key=lambda x: x[0])
        
        for dist, ret_code in results_by_record[record_name]:
            dist_str = f"{dist:.1f}"
            if ret_code == 0:
                print(f"[{record_name}] Distance {dist_str}m: SUCCESS (mani_succ=True)")
                success_count += 1
                total_success += 1
            elif ret_code == 2:
                print(f"[{record_name}] Distance {dist_str}m: SKIPPED (no prediction)")
                skip_count += 1
                total_skip += 1
            elif ret_code == 3:
                print(f"[{record_name}] Distance {dist_str}m: FAILED (mani_succ=False)")
                fail_count += 1
                total_fail += 1
            elif ret_code == 4:
                print(f"[{record_name}] Distance {dist_str}m: SUCCESS (mani_succ=True, already exists)")
                success_count += 1
                total_success += 1
                existing_count += 1
                total_existing += 1
            elif ret_code == 5:
                print(f"[{record_name}] Distance {dist_str}m: FAILED (mani_succ=False, already exists)")
                fail_count += 1
                total_fail += 1
                existing_count += 1
                total_existing += 1
                existing_count += 1
                total_existing += 1
            else:
                print(f"[{record_name}] Distance {dist_str}m: FAILED (execution error)")
                fail_count += 1
                total_fail += 1
        
        print(f"\n[{record_name}] === Summary ===")
        print(f"Success (mani_succ=True): {success_count}")
        print(f"Failed (mani_succ=False or error): {fail_count}")
        print(f"Skipped (no prediction): {skip_count}")
        if existing_count > 0:
            print(f"Already existed (skipped execution): {existing_count}")
        print(f"Total distances: {len(distances)}")
        tested_count = len(distances) - skip_count
        if tested_count > 0:
            success_rate = success_count / tested_count * 100
            print(f"Success rate: {success_rate:.1f}% ({success_count}/{tested_count})")
        else:
            print("Success rate: N/A (no distances tested)")

    print(f"\n{'='*60}")
    print(f"=== Overall Results (all samples) ===")
    print(f"Total samples processed: {len(valid_record_names)}")
    print(f"Total success (mani_succ=True): {total_success}")
    print(f"Total failed (mani_succ=False or error): {total_fail}")
    print(f"Total skipped (no prediction): {total_skip}")
    if total_existing > 0:
        print(f"Total already existed (skipped execution): {total_existing}")
    total_tested = total_success + total_fail
    if total_tested > 0:
        overall_success_rate = total_success / total_tested * 100
        print(f"Overall success rate: {overall_success_rate:.1f}% ({total_success}/{total_tested})")
    else:
        print("Overall success rate: N/A (no distances tested)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

