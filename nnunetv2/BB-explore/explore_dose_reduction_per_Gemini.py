1+1

import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from PIL import Image

# Import necessary transforms from batchgenerators
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.abstract_transforms import Compose

class PoissonNoiseTransform(AbstractTransform):
    def __init__(self, p_per_sample=1.0):
        self.p_per_sample = p_per_sample
    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            data = data_dict['data']
            # Scale data to expected photon counts (assuming values are in [0,1])
            # You can adjust this scaling factor based on your needs
            scale = 200.0  # Reduced from 300.0 to increase graininess further
            scaled_data = data * scale
            
            # Add Poisson noise
            noisy_data = np.random.poisson(scaled_data)
            
            # Scale back to original range
            data_dict['data'] = noisy_data / scale
        return data_dict

# --- Configuration ---
CONVENTIONAL_DOSE_PATH = "/Users/billb/Desktop/AnnotationRevisions/CineAngio.png"
LOW_DOSE_PATH = "/Users/billb/Desktop/AnnotationRevisions/FluoroAngio.png"

# --- Check if files exist ---
if not os.path.exists(CONVENTIONAL_DOSE_PATH):
    print(f"Error: Conventional dose image not found at {CONVENTIONAL_DOSE_PATH}")
    exit()
if not os.path.exists(LOW_DOSE_PATH):
    print(f"Error: Low dose image not found at {LOW_DOSE_PATH}")
    exit()

# --- Load images ---
conventional_img = np.array(Image.open(CONVENTIONAL_DOSE_PATH))
low_dose_img = np.array(Image.open(LOW_DOSE_PATH))

# --- Prepare data for batchgenerators ---
# Convert to float and normalize to [0, 1] range
conventional_img_float = conventional_img.astype(np.float32) / 255.0
low_dose_img_float = low_dose_img.astype(np.float32) / 255.0

# Add batch and channel dimensions: (height, width) -> (1, 1, height, width)
conventional_img_batch = conventional_img_float[np.newaxis, np.newaxis, :, :]

# --- Define transforms ---
# Combined transform for low dose simulation
transform_low_dose_sim = Compose([
    ContrastAugmentationTransform(contrast_range=(0.2, 0.4), p_per_sample=1.0, preserve_range=True),
    PoissonNoiseTransform(p_per_sample=1.0),
    GaussianNoiseTransform(noise_variance=(0.05, 0.1), p_per_sample=1.0)
])

# --- Apply transform to conventional dose image ---
data_dict = {'data': conventional_img_batch}
simulated_low_dose_dict = transform_low_dose_sim(**data_dict)
simulated_low_dose = simulated_low_dose_dict['data'][0, 0, :, :]

# --- Display results ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Display original conventional dose image
im0 = axes[0].imshow(conventional_img_float, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Conventional Dose")
axes[0].axis('off')

# Display simulated low dose
im1 = axes[1].imshow(simulated_low_dose, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Simulated Low Dose")
axes[1].axis('off')

# Display actual low dose
im2 = axes[2].imshow(low_dose_img_float, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Actual Low Dose")
axes[2].axis('off')

plt.tight_layout()
plt.show()