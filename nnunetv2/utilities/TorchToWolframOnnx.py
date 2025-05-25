1+1

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import copy
import dill

import sys
sys.path.append('/Users/billb/github/nnUNet-Adjustment')
import nnunetv2.training.nnUNetTrainer

from typing import Dict, Any, Tuple, List, Optional, Union

def fix_normalization_for_onnx_export(model: nn.Module) -> nn.Module:
    """
    Fix BatchNorm and InstanceNorm layers for consistent ONNX export behavior,
    ensuring they work properly in inference mode with Wolfram compatibility.
    
    Args:
        model: The PyTorch model to fix
        
    Returns:
        The fixed model ready for export
    """
    # Create a deep copy to avoid modifying the original
    model_copy = copy.deepcopy(model)
    model_copy.eval()  # Ensure model is in evaluation mode
    
    # Fix BatchNorm and InstanceNorm layers
    for module in model_copy.modules():
        # Fix BatchNorm layers
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Ensure running stats are tracked
            module.track_running_stats = True
            
            # Initialize running stats if they don't exist
            if module.running_mean is None:
                module.running_mean = torch.zeros(
                    module.num_features, device=module.weight.device if hasattr(module, 'weight') else 'cpu'
                )
            if module.running_var is None:
                module.running_var = torch.ones(
                    module.num_features, device=module.weight.device if hasattr(module, 'weight') else 'cpu'
                )
                
            # Ensure the module is in eval mode
            module.training = False
        
        # Fix InstanceNorm layers
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            # For InstanceNorm, we need to ensure track_running_stats is True
            # This is crucial for consistent behavior across batch elements
            module.track_running_stats = True
            
            # Initialize running stats if they don't exist
            if not hasattr(module, 'running_mean') or module.running_mean is None:
                module.running_mean = torch.zeros(
                    module.num_features, device=module.weight.device if hasattr(module, 'weight') else 'cpu'
                )
            if not hasattr(module, 'running_var') or module.running_var is None:
                module.running_var = torch.ones(
                    module.num_features, device=module.weight.device if hasattr(module, 'weight') else 'cpu'
                )
            
            # Ensure the module is in eval mode
            module.training = False

    return model_copy

def replace_instance_norm_with_batch_norm(model: nn.Module) -> nn.Module:
    """
    Replace InstanceNorm with equivalent BatchNorm for better compatibility.
    This approach works when batch size is fixed to 1 for export.
    
    Args:
        model: The PyTorch model to modify
        
    Returns:
        Modified model with InstanceNorm layers replaced
    """
    # Create a deep copy to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Track which modules to replace
    replacements = {}
    
    # Identify InstanceNorm modules to replace
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            # Create equivalent BatchNorm
            if isinstance(module, nn.InstanceNorm1d):
                replacement = nn.BatchNorm1d(
                    module.num_features,
                    eps=module.eps,
                    momentum=0.1,  # Default BatchNorm momentum
                    affine=module.affine,
                    track_running_stats=True
                )
            elif isinstance(module, nn.InstanceNorm2d):
                replacement = nn.BatchNorm2d(
                    module.num_features,
                    eps=module.eps,
                    momentum=0.1,
                    affine=module.affine,
                    track_running_stats=True
                )
            elif isinstance(module, nn.InstanceNorm3d):
                replacement = nn.BatchNorm3d(
                    module.num_features,
                    eps=module.eps,
                    momentum=0.1,
                    affine=module.affine,
                    track_running_stats=True
                )
            
            # Copy affine parameters if they exist
            if module.affine and hasattr(module, 'weight') and hasattr(module, 'bias'):
                replacement.weight.data.copy_(module.weight.data)
                replacement.bias.data.copy_(module.bias.data)
            
            # Initialize running stats
            replacement.running_mean.zero_()
            replacement.running_var.fill_(1)
            replacement.training = False
            
            replacements[name] = replacement
    
    # Perform the replacements
    for name, replacement in replacements.items():
        # Split name into parent module and attribute
        parts = name.split('.')
        parent_name = '.'.join(parts[:-1])
        attr_name = parts[-1]
        
        # Get the parent module
        parent = model_copy
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, attr_name, replacement)
    
    return model_copy

def create_wolfram_compatible_model(model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
    """
    Prepare a model for export to ONNX with Wolfram compatibility,
    handling batch normalization and instance normalization issues.
    
    Args:
        model: The original PyTorch model
        input_shape: The input shape (excluding batch dimension)
        
    Returns:
        A Wolfram-compatible model
    """
    # First fix the normalization layers
    model = fix_normalization_for_onnx_export(model)
    
    # Option: Replace InstanceNorm with BatchNorm (uncomment if needed)
    # model = replace_instance_norm_with_batch_norm(model)
    
    # Create a dummy input with batch size 1 to run a forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)
    
    # Run a forward pass to ensure everything is initialized
    with torch.no_grad():
        model(dummy_input)
    
    return model

def export_model_for_wolfram(
    model: nn.Module, 
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 11,
    verify: bool = True
) -> str:
    """
    Export a PyTorch model to ONNX format optimized for Wolfram import.
    
    Args:
        model: The PyTorch model to export
        input_shape: The input shape (excluding batch dimension)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
        verify: Whether to verify the export with ONNX Runtime
        
    Returns:
        Path to the exported model
    """
    # Prepare the model
    model = create_wolfram_compatible_model(model, input_shape)
    
    # Create a dummy input with batch size 1
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # No dynamic_axes to enforce fixed batch size of 1
    )
    
    # Verify the model structure with ONNX checker
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model structure verified and saved to {output_path}")
    
    # Verify export results match
    if verify:
        verify_onnx_export(model, output_path, dummy_input)
    
    return output_path

def verify_onnx_export(
    pytorch_model: nn.Module,
    onnx_path: str,
    test_input: torch.Tensor
) -> bool:
    """
    Verify that the exported ONNX model produces the same outputs as PyTorch.
    
    Args:
        pytorch_model: The PyTorch model
        onnx_path: Path to the exported ONNX model
        test_input: Input tensor for testing
        
    Returns:
        True if verification passes
    """
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        torch_output = pytorch_model(test_input).cpu().numpy()
    
    # Get ONNX Runtime output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    try:
        np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)
        print("✓ Export validation successful: PyTorch and ONNX outputs match!")
        return True
    except AssertionError as e:
        print(f"✗ Export validation failed: {e}")
        print(f"Max absolute difference: {np.max(np.abs(torch_output - ort_outputs[0]))}")
        return False

# Example usage:
if __name__ == "__main__":
    # Example model with both BatchNorm and InstanceNorm
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Model with both types of normalization
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.in1 = nn.InstanceNorm2d(32)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32 * 32 * 32, 10)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.in1(self.conv2(x)))
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    # Create model
    # model = ExampleModel()

gpuDevice = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

torchModelPath =  "/Volumes/X10Pro/AWIBuffer/UXlstmBot-nnUNetPlans_2d-reduced3-DC_and_CE_loss-w-1-20-40-dill.pth"

model = torch.load(torchModelPath, map_location=gpuDevice, weights_only=False)
    
    # Export for Wolfram
    export_model_for_wolfram(
        model=model,
        input_shape=(3, 32, 32),  # C, H, W 
        output_path="wolfram_compatible_model.onnx",
        opset_version=11
    )