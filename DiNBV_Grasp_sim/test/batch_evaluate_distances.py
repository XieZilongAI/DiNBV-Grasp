"""
Batch evaluate all distances (1m-6m) for test_data_camera_distances
Generate labels for training distance selector
"""
from argparse import ArgumentParser
import torch
import llama
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from subprocess import call
import time

def process_samples_on_gpu(gpu_id, sample_list, adapter_dir, llama_dir, data_dir, output_dir, prompt, distances):
    """Process a list of samples on a specific GPU"""
    # Set the current GPU device before loading model
    # When CUDA_VISIBLE_DEVICES is set, gpu_id 0 maps to the first visible GPU, etc.
    if torch.cuda.is_available():
        # Force set the device before any CUDA operations
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        # Verify we're on the correct device
        current_device = torch.cuda.current_device()
        print(f"Process on GPU {gpu_id} (device: {device}, current_device: {current_device}) starting, will process {len(sample_list)} samples")
    else:
        device = torch.device('cpu')
        print(f"Process on CPU starting, will process {len(sample_list)} samples")
    
    # Load model on this GPU - ensure checkpoint is loaded to CPU first, then moved to target device
    load_result = llama.load(adapter_dir, llama_dir, str(device))
    if isinstance(load_result, RuntimeError):
        print(f"Error loading model on GPU {gpu_id}: {load_result}")
        return
    model, preprocess = load_result
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    # Clear cache to free up any unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Print memory info for debugging
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU {gpu_id} memory after model load: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Process samples assigned to this GPU
    for sample_name in tqdm(sample_list, desc=f"GPU {gpu_id}"):
        sample_dir = os.path.join(data_dir, sample_name)
        sample_out_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_out_dir, exist_ok=True)
        
        # Evaluate each distance
        for dist in distances:
            dist_str = f"{dist:.1f}"
            image_path = os.path.join(sample_dir, f'camera_rgb_dist{dist_str}m.png')
            
            if not os.path.exists(image_path):
                continue
            
            # Output path for this distance
            output_json = os.path.join(sample_out_dir, f'pred_dist{dist_str}.json')
            
            if os.path.exists(output_json):
                continue  # Skip if already evaluated
            
            # Load and preprocess image
            img_pil = Image.open(image_path).convert('RGB')
            img_array = np.array(img_pil)[:336, :336, :]
            img_pil = Image.fromarray(img_array)
            # Ensure image tensor is created on the correct device
            img = preprocess(img_pil).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                result = model.generate(img, [prompt])[0]
            
            # Save prediction
            with open(output_json, 'w') as fout:
                json.dump(result, fout)
            
            # Clear cache periodically to avoid memory buildup
            if torch.cuda.is_available() and (dist == distances[-1] or (len(distances) > 10 and int(dist * 10) % 10 == 0)):
                torch.cuda.empty_cache()

def main():
    parser = ArgumentParser()
    parser.add_argument('--llama_dir', type=str,
                        default='/data1/xzl/ManipLLM/train/ckpts/llama_model_weights',
                        help='llama model directory')
    parser.add_argument('--adapter_dir', type=str,
                        default='/data1/xzl/ManipLLM/train/exp/train_ckpts_dist5.0/checkpoint-9.pth',
                        help='adapter checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='../data_collection/data/test_data_camera_distances',
                        help='test_data_camera_distances directory')
    parser.add_argument('--output_dir', type=str,
                        default='./distance_eval_results',
                        help='output directory for evaluation results')
    parser.add_argument('--num_processes', type=int, default=10, help='number of parallel processes')
    parser.add_argument('--num_gpus', type=int, default=None, help='number of GPUs to use (default: all available)')
    parser.add_argument('--action', type=str, default='pulling')
    parser.add_argument('--eval_in_sim', action='store_true', help='evaluate in SAPIEN simulator')
    args = parser.parse_args()

    # Step 1: Model inference for all distances
    print("Step 1: Running model inference on all distances...")

    # Determine number of GPUs to use
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_gpus = args.num_gpus

    print(f"Using {num_gpus} GPU(s) for parallel processing")

    if '-ori' in args.adapter_dir:
        prompt = llama.format_prompt('Specify the contact point and orientation of pushing the object.')
    else:
        prompt = llama.format_prompt('Specify the contact point and gripper direction of manipulating the object.')

    # Get all sample folders
    sample_folders = [f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))]

    # Distances to evaluate
    distances = np.arange(1.0, 6.05, 0.1)

    # Split samples across GPUs
    if num_gpus > 1:
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Distribute samples across GPUs
        samples_per_gpu = len(sample_folders) // num_gpus
        processes = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * samples_per_gpu
            if gpu_id == num_gpus - 1:
                # Last GPU gets remaining samples
                gpu_samples = sample_folders[start_idx:]
            else:
                gpu_samples = sample_folders[start_idx:start_idx + samples_per_gpu]
            
            if len(gpu_samples) == 0:
                continue
                
            p = mp.Process(target=process_samples_on_gpu, 
                          args=(gpu_id, gpu_samples, args.adapter_dir, args.llama_dir, 
                                args.data_dir, args.output_dir, prompt, distances))
            p.start()
            processes.append(p)
            # Small delay to avoid simultaneous resource competition during model loading
            time.sleep(2)  # Wait 2 seconds before starting next process
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"All GPU processes completed.")
    else:
        # Single GPU or CPU mode
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        load_result = llama.load(args.adapter_dir, args.llama_dir, device)
        if isinstance(load_result, RuntimeError):
            print(f"Error loading model: {load_result}")
            exit(1)
        model, preprocess = load_result
        model.to(device)
        model.eval()
        
        for sample_name in tqdm(sample_folders, desc="Processing samples"):
            sample_dir = os.path.join(args.data_dir, sample_name)
            sample_out_dir = os.path.join(args.output_dir, sample_name)
            os.makedirs(sample_out_dir, exist_ok=True)
            
            # Evaluate each distance
            for dist in distances:
                dist_str = f"{dist:.1f}"
                image_path = os.path.join(sample_dir, f'camera_rgb_dist{dist_str}m.png')
                
                if not os.path.exists(image_path):
                    continue
                
                # Output path for this distance
                output_json = os.path.join(sample_out_dir, f'pred_dist{dist_str}.json')
                
                if os.path.exists(output_json):
                    continue  # Skip if already evaluated
                
                # Load and preprocess image
                img_pil = Image.open(image_path).convert('RGB')
                img_array = np.array(img_pil)[:336, :336, :]
                img_pil = Image.fromarray(img_array)
                img = preprocess(img_pil).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    result = model.generate(img, [prompt])[0]
                
                # Save prediction
                with open(output_json, 'w') as fout:
                    json.dump(result, fout)

    print(f"Step 1 complete. Predictions saved to {args.output_dir}")

    # Step 2: Execute in simulator if requested
    if args.eval_in_sim:
        print("\nStep 2: Evaluating in SAPIEN simulator...")
        print("This will test each predicted grasp in simulation to get success/failure labels")
        print(f"Run: python test_entireprocess_in_sapien.py --data_dir {args.data_dir} --out_dir {args.output_dir} --num_processes {args.num_processes} --no_gui")
    else:
        print("\nSkipping simulator evaluation. Use --eval_in_sim to evaluate in SAPIEN")

    print("\nDone! Evaluation results saved.")


if __name__ == '__main__':
    main()



