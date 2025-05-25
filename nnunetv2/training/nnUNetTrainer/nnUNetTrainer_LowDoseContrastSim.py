import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

# At deployment:
# nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD -tr nnUNetTrainer_LowDoseContrastSim

class PoissonNoiseTransform(AbstractTransform):
    def __init__(self, p_per_sample=1.0):
        self.p_per_sample = p_per_sample
    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            data = data_dict['image']
            # Convert to numpy if it's a tensor
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            
            # Ensure data is positive and finite
            data = np.clip(data, 0, None)  # Clip negative values to 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and infinities
            
            # Scale the data to a reasonable range for Poisson noise
            scale = 200.0  # Matches our tested value for maximum graininess
            scaled_data = data * scale
            
            # Generate Poisson noise
            noisy_data = np.random.poisson(scaled_data)
            
            # Scale back and ensure valid range
            noisy_data = np.clip(noisy_data / scale, 0, 1)
            
            # Convert back to tensor with the same device as input
            if torch.is_tensor(data_dict['image']):
                noisy_data = torch.from_numpy(noisy_data).to(data_dict['image'].device)
            
            data_dict['image'] = noisy_data
        return data_dict

class nnUNetTrainer_LowDoseContrastSim(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Using Custom Trainer nnUNetTrainer_LowDoseContrastSim with 30% low dose simulation")

    def get_training_transforms(self, *args, **kwargs):
        # Get the default augmentations from the parent class
        base_transforms = super().get_training_transforms(*args, **kwargs)

        # Define low dose simulation transforms using our tested parameters
        low_dose_sim = Compose([
            ContrastAugmentationTransform(contrast_range=(0.2, 0.4), p_per_sample=1.0, preserve_range=True, data_key='image'),
            PoissonNoiseTransform(p_per_sample=1.0),
            GaussianNoiseTransform(noise_variance=(0.05, 0.1), p_per_sample=1.0, data_key='image')
        ])

        # Add low dose simulation to 30% of batches
        # We'll use a random transform that applies the low dose simulation with 30% probability
        class RandomLowDoseTransform(AbstractTransform):
            def __init__(self, low_dose_transform, p=0.3):
                self.low_dose_transform = low_dose_transform
                self.p = p

            def __call__(self, **data_dict):
                if np.random.uniform() < self.p:
                    return self.low_dose_transform(**data_dict)
                return data_dict

        # Create a new ComposeTransforms with both the base transforms and our low dose transform
        all_transforms = ComposeTransforms([
            base_transforms,
            RandomLowDoseTransform(low_dose_transform=low_dose_sim, p=0.3)
        ])

        return all_transforms

    # You might need to override other DA-related methods depending on
    # exactly how augmentation is structured in your nnUNet version and base class.
    # Check `nnUNetTrainerV2.initialize_data_augmentation` and related methods.