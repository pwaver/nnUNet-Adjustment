import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DiceFocalBCELoss  # Import the custom loss
from nnunetv2.training.loss.dice_focal_loss import DiceLoss  # Add this import

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        # Initialize alpha based on class frequencies
        self.alpha = alpha if alpha is not None else torch.tensor([1.0, 10.0, 30.0])
        self.alpha = self.alpha / self.alpha.sum()  # Normalize

    def forward(self, inputs, targets):
        """
        inputs: Logits from model (N, C, H, W)
        targets: Ground truth masks (N, 1, H, W)
        """
        ce_loss = F.cross_entropy(inputs, targets.squeeze(1).long(), 
                                weight=self.alpha.to(inputs.device),
                                reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=0.7, dice_weight=0.3):
        """
        Initialize the combined loss with configurable weights
        
        Args:
            alpha: Class weights for focal loss
            gamma: Focal loss focusing parameter
            focal_weight: Weight for focal loss component (default: 0.7)
            dice_weight: Weight for dice loss component (default: 0.3)
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = MultiClassFocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        return (self.focal_weight * self.focal_loss(inputs, targets) + 
                self.dice_weight * self.dice_loss(inputs, targets))

class nnUNetTrainerFocalDiceBCELoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: str = "cuda"
    ) -> None:
        """Initialize the FocalDiceBCELoss trainer.
        
        Args:
            plans: nnUNet plans dictionary
            configuration: Configuration to use
            fold: Current fold in cross-validation
            dataset_json: Dataset JSON configuration
            device: Device to use for training
        """
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device
        )
        # Initialize loss weights
        self.focal_weight = 0.7
        self.dice_weight = 0.3
        self.alpha = torch.tensor([1.0, 10.0, 30.0])
        self.gamma = 2.0
        # Your custom initialization code here
        self.loss = self.get_loss()
        
    def get_loss(self):
        """Get the combined Focal, Dice, and BCE loss function."""
        from torch import nn
        import torch
        
        class FocalDiceBCELoss(nn.Module):
            def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
                super().__init__()
                self.focal = MultiClassFocalLoss(gamma=gamma)
                self.dice = DiceLoss()
                self.bce = nn.BCEWithLogitsLoss()
                self.alpha = alpha

            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                focal_loss = self.focal(input, target)
                dice_loss = self.dice(input, target)
                bce_loss = self.bce(input, target)
                return focal_loss + dice_loss + bce_loss

        return FocalDiceBCELoss()

    def _build_loss(self):
        self.print_to_log_file(f"Using Dice + BCE + Focal Loss for training with weights: "
                              f"focal={self.focal_weight}, dice={self.dice_weight}")

        # Define the loss function with configurable parameters
        loss = CombinedLoss(
            alpha=self.alpha,
            gamma=self.gamma,
            focal_weight=self.focal_weight,
            dice_weight=self.dice_weight
        )

        # Apply Deep Supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # Avoid errors with Distributed Data Parallel (DDP)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6  # Prevent unused parameter issues
            else:
                weights[-1] = 0  # Last output has no weight

            # Normalize weights
            weights = weights / weights.sum()

            # Wrap with Deep Supervision
            loss = DeepSupervisionWrapper(loss, weights)

        return loss