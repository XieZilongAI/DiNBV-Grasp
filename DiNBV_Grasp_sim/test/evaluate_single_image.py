"""
Evaluate a single image for grasp prediction
Used for evaluating images at different camera distances
"""
from argparse import ArgumentParser
import torch
import llama
import os
from PIL import Image
import json
import numpy as np

parser = ArgumentParser()
parser.add_argument('--llama_dir', type=str, 
                    default='/data1/xzl/ManipLLM/train/ckpts/llama_model_weights',
                    help='llama model directory')
parser.add_argument('--adapter_dir', type=str,
                    default='/data1/xzl/ManipLLM/train/exp/train_ckpts_dist5.0/checkpoint-9.pth',
                    help='adapter checkpoint directory')
parser.add_argument('--image_path', type=str, default='../data_collection/data/test_data_camera_distances/149_Faucet_3_pulling_5/camera_rgb_dist1.9m.png', help='path to input RGB image')
parser.add_argument('--output_path', type=str, default='./test_results/just_for_test/test1.json', help='path to save prediction result')
parser.add_argument('--action', type=str, default='pulling', help='action type')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
print(f"Loading model from {args.adapter_dir} and {args.llama_dir}")
model, preprocess = llama.load(args.adapter_dir, args.llama_dir, device)
model.to(device)
model.eval()
print("Model loaded successfully")
# Prepare prompt
if '-ori' in args.adapter_dir:
    prompt = llama.format_prompt('Specify the contact point and orientation of pushing the object.')
else:
    prompt = llama.format_prompt('Specify the contact point and gripper direction of manipulating the object.')

# Load and preprocess image
img_pil = Image.open(args.image_path).convert('RGB')
img_array = np.array(img_pil)[:336, :336, :]  # Crop to 336x336
img_pil = Image.fromarray(img_array)
img = preprocess(img_pil).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    result = model.generate(img, [prompt])[0]

# Save prediction
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w') as fout:
    json.dump(result, fout)

print(f"Prediction saved to {args.output_path}")



