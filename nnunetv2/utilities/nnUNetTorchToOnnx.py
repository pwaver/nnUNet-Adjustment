

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import copy
import dill
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/billb/github/nnUNet-Adjustment')

# Set up paths
is_mac = os.name == 'posix' and os.uname().sysname == 'Darwin'
rootPath = "/Volumes/X10Pro/AWIBuffer/nnUNet/data/" if is_mac else '/mnt/SliskiDrive/AWI/AWIBuffer/nnUNet' 

# Set up nnUNet environment variables to avoid path errors
os.environ['nnUNet_raw'] = join(rootPath, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = join(rootPath, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = join(rootPath, 'nnUNet_results')

# nnUNet imports
import nnunetv2.training.nnUNetTrainer
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LowDoseContrastSim
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from typing import Dict, Any, Tuple, List, Optional, Union

gpuDevice = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

torchModelPath =  "/Volumes/X10Pro/AWIBuffer/UXlstmBot-nnUNetPlans_2d-reduced3-DC_and_CE_loss-w-1-20-40-dill.pth"
torchModelPath =  "/Volumes/X10Pro/AWIBuffer/nnUNet/data/nnUNet_results/Dataset332_Angiography/nnUNetTrainer_LowDoseContrastSim__nnUNetResEncUNetLPlans__2d/fold_all/checkpoint_final.pth"

# instantiate the nnUNetPredictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

model_training_output_dir = '/Volumes/X10Pro/AWIBuffer/nnUNet/data/nnUNet_results/Dataset332_Angiography/nnUNetTrainer_LowDoseContrastSim__nnUNetResEncUNetLPlans__2d/'
checkpoint_name = 'checkpoint_final.pth'

# initializes the network architecture, loads the checkpoint

predictor.initialize_from_trained_model_folder(
    '/Volumes/X10Pro/AWIBuffer/nnUNet/data/nnUNet_results/Dataset332_Angiography/nnUNetTrainer_LowDoseContrastSim__nnUNetResEncUNetLPlans__2d/',
    use_folds='all')

model = predictor.network
# model.eval()

# model = torch.load(torchModelPath, map_location=gpuDevice, weights_only=False)

# Handle checkpoint format - extract the model if it's a dictionary
if isinstance(model, dict):
    if 'network_weights' in model:
        # This is a nnUNet checkpoint format
        print("Loading from nnUNet checkpoint format")
        # You'll need to reconstruct the model architecture here
        # For now, skip this complex case
        print("Checkpoint loading not implemented - using direct model path")
        exit()
    elif 'model_state_dict' in model:
        # Standard PyTorch checkpoint
        model = model['model_state_dict']
    else:
        print("Unknown checkpoint format")
        print("Available keys:", model.keys())
        exit()
    
model.eval()

# Move model to the correct device
model = model.to(gpuDevice)
print(f"Model moved to device: {gpuDevice}")

# Test the model with a random input
random_tensor = torch.randn(1, 5, 512, 512, device=gpuDevice, dtype=torch.float32)
print("Input tensor shape:", random_tensor.shape)

with torch.inference_mode():
    output = model(random_tensor)

print("Output tensor shape:", output.shape)

import platform

system = platform.system()
if "Darwin" in system:
    if os.path.isdir("/Volumes/X10Pro"):
        dataDir = "/Volumes/X10Pro/AWIBuffer"
    else:
        dataDir = "/Users/billb/Projects/AWI/NetExploration"
elif "Linux" in system:
    dataDir = "/mnt/SliskiDrive/AWI/AWIBuffer"
else:
    dataDir = None  # or some default path

angiogramH5Path = dataDir + "/WebknossosAngiogramsRevisedUInt8List.h5"
angiogramH5Path

import h5py

# Open the HDF5 file and print all dataset keys
with h5py.File(angiogramH5Path, 'r') as f:
    # Get all keys at root level
    keys = list(f.keys())
    print("Dataset keys in HDF5 file:")
    for key in keys:
        print(f"- {key}")

# Load first angiogram from HDF5 file
import random
with h5py.File(angiogramH5Path, 'r') as f:
    # Get first key
    hdfKey = random.choice(keys)
    print(f"Loading dataset: {hdfKey}")
    # Load data into tensor
    agram = torch.from_numpy(f[hdfKey][:]).float()
    print(f"Loaded tensor shape: {agram.shape}")

#Display the 30th frame of the angiogram
plt.imshow(agram[30], cmap='gray')
plt.colorbar()
plt.show()

# Normalize angiogram by subtracting mean and dividing by standard deviation
xagram = (agram - agram.mean()) / agram.std()
print(f"Normalized tensor shape: {xagram.shape}")

# Create input tensor with 5 consecutive frames centered around frame 30
start_idx = 28  # 30-2 to get 2 frames before
end_idx = 33    # 30+3 to get 2 frames after (exclusive)
z = xagram[start_idx:end_idx].unsqueeze(0)  # Add batch dimension
print(f"Input tensor shape: {z.shape}")

z = z.to(gpuDevice)
y=model(z)
y.shape
# Apply softmax along dimension 1 (second dimension) which has size 3
y = torch.nn.functional.softmax(y, dim=1)
print(f"Output tensor shape after softmax: {y.shape}")
# Display the 3rd channel (index 2) of the output
plt.imshow(y[0, 2].cpu().detach().numpy(), cmap='gray')
plt.colorbar()
plt.title('Output Channel 3')
plt.show()

# Calculate number of valid frame groups (each group has 5 consecutive frames)
num_frames = xagram.shape[0]
num_groups = num_frames - 4  # Each group needs 5 frames

# Create tensor to hold all valid frame groups
z5 = torch.zeros((num_groups, 5, 512, 512))

# Fill z5 with overlapping groups of 5 consecutive frames
for i in range(num_groups):
    z5[i] = xagram[i:i+5]

print(f"Shape of tensor containing all valid 5-frame groups: {z5.shape}")

# Feed z5 into the model and get the output
y5 = model(z5.to(gpuDevice))
y5.shape

# Apply softmax along dimension 1 (second dimension) which has size 3
ys5 = torch.nn.functional.softmax(y5, dim=1)
print(f"Output tensor shape after softmax: {ys5.shape}")

# Display the 3rd channel (index 2) of batch member 35
plt.imshow(ys5[35, 2].cpu().detach().numpy(), cmap='gray')
plt.colorbar()
plt.title('Output Channel 3 - Batch 35')
plt.show()

# Export model back to ONNX
onnxOutputPath = torchModelPath.replace(".pth", ".onnx")

with torch.inference_mode():
    torch.onnx.export(
    model,
    z,
    onnxOutputPath,
    export_params=True,
    opset_version=18,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    keep_initializers_as_inputs=True,  # This can help with some batch dimension issues
    do_constant_folding=True
)

torch.onnx.export(
    model,
    z,
    onnxOutputPath,
    export_params=True,
    opset_version=18,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    keep_initializers_as_inputs=True,  # This can help with some batch dimension issues
    do_constant_folding=True
)
