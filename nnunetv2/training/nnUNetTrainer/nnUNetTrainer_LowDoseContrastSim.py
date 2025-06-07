import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from nnunetv2.training.data_augmentation.custom_transforms.LowDoseTransform import (
    PoissonNoiseTransform, 
    RandomLowDoseTransform
)

# At deployment:
# nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD -tr nnUNetTrainer_LowDoseContrastSim



class nnUNetTrainer_LowDoseContrastSim(nnUNetTrainer):
    """nnUNet trainer with low dose contrast simulation for fluoroscopic cardiac angiography.
    
    This trainer adds data augmentation that simulates physically low radiation dose images
    during training by applying contrast reduction, Poisson noise, and Gaussian noise to
    30% of training batches.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')) -> None:
        """Initialize the low dose contrast simulation trainer.
        
        Args:
            plans: Training plans dictionary.
            configuration: Configuration name.
            fold: Cross-validation fold number.
            dataset_json: Dataset configuration dictionary.
            device: Torch device for training. Defaults to CUDA.
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Using Custom Trainer nnUNetTrainer_LowDoseContrastSim with 30% low dose simulation")

    def get_training_transforms(self, *args, **kwargs) -> BasicTransform:
        """Get training transforms including low dose simulation.
        
        Extends the base training transforms by adding low dose contrast simulation
        that includes contrast reduction, Poisson noise, and Gaussian noise. The
        low dose simulation is applied to 30% of training batches.
        
        Args:
            *args: Variable length argument list passed to parent method.
            **kwargs: Arbitrary keyword arguments passed to parent method.
            
        Returns:
            Combined transform including base augmentations and low dose simulation.
        """
        # Get the default augmentations from the parent class
        base_transforms = super().get_training_transforms(*args, **kwargs)

        # Define low dose simulation transforms using v2 transforms with tested parameters
        low_dose_sim = ComposeTransforms([
            ContrastTransform(
                contrast_range=BGContrast((0.2, 0.4)), 
                preserve_range=True, 
                synchronize_channels=False,
                p_per_channel=1
            ),
            PoissonNoiseTransform(p_per_sample=1.0),
            GaussianNoiseTransform(
                noise_variance=(0.05, 0.1), 
                p_per_channel=1,
                synchronize_channels=True
            )
        ])

        # Add low dose simulation to 30% of batches using the imported RandomLowDoseTransform
        random_low_dose = RandomLowDoseTransform(low_dose_transform=low_dose_sim, p=0.3)

        # Create a new ComposeTransforms with both the base transforms and our low dose transform
        all_transforms = ComposeTransforms([
            base_transforms,
            random_low_dose
        ])

        return all_transforms
