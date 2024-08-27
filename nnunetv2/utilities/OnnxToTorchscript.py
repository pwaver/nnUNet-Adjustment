1+1
import torch
import onnx
import dill
from onnx2torch import convert

import os
rootPath = '/mnt/SliskiDrive/AWI/AWIBuffer/' if os.name == 'posix' else '/Volumes/Crucial X8/AWI/Data/'

onnxPath = rootPath + "UMambaBot-plans_unet_edge8_epochs250_2d-DC_and_CE_loss-w-1-20-20.onnx"

modelPerOnnx = convert(onnxPath)

gpuDevice = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

modelPerOnnx = modelPerOnnx.to(gpuDevice)

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape
random_tensor = random_tensor.to(gpuDevice)

result = modelPerOnnx(random_tensor)
result.shape

# Scripted does not work:
# scriptedModelperOnnx = torch.jit.script(modelPerOnnx)
# scripted_model.save("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

# Try traced # update: traced after re-import from onnx seems to work
tracedModelperOnnx = torch.jit.trace(modelPerOnnx, random_tensor)
# Error types: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
# TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results)
# TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.

tracedModelPath = onnxPath.replace(".onnx", "-torchscript-traced-onnx.pt")

tracedModelperOnnx.save(tracedModelPath)

# Read back in traced model path and see if works

# tracedModelPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced-onnx.pt"

tracedModel = torch.jit.load(tracedModelPath)

tracedModel = tracedModel.to(gpuDevice)

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape
random_tensor = random_tensor.to(gpuDevice)

result = tracedModel(random_tensor)
result.shape