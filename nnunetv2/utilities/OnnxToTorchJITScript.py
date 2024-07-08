# This script is used to convert the ONNX model to a TorchScript model
# The ONNX model is converted to a TorchScript model using the ONNX-Torch JIT compiler
# The TorchScript model is then saved to a file
# import modules

# NB fails on Mac, reporting: incompatible architecture (have 'arm64', need 'x86_64')
1+1
import onnx
import torch
from onnx2torch import convert

onnx_model_path = "/Users/billb/Projects/AWI/NetExploration/UMambaBot-nnUNetPlans_2d-edge8-w-1-10-10.onnx"

torch_model_1 = convert(onnx_model_path)

