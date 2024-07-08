# import all the necessary libraries to read a PyTorch model exported with dill, add a softmax layer and save it again with dill
1+1
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import dill
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data
import nnunetv2
import h5py

angiogramsH5Path="/Users/billb/Projects/AWI/NetExploration/AngiogramsUInt8List.h5"

with h5py.File(angiogramsH5Path, 'r') as fr:
    data=np.array(fr["30"])

# function to z-normalize data
def zNormalizeArray(arr):
    # Calculate the mean and standard deviation along the entire array
    mean = np.mean(arr)
    std = np.std(arr) + .0001 # protect from instability if std is too small.    
    return (arr - mean) / std

# divvy into groups of 5
# data = np.array_split(data, 5)
zdata = np.array([zNormalizeArray(data[n-2:n+3]) for n in range(2,len(data)-2)])

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU.")

modelPath = "/Users/billb/Projects/AWI/NetExploration/UNETR-nnUNetPlans_2d-DC_and_CE_loss-w-1-20-20-dill.pth"
modelPath = "/Users/billb/Projects/AWI/NetExploration/UMambaBot-plans_unet_edge8_features256_2d-DC_and_CE_loss-w-1-20-20-dill.pth"

# read the model using dill
model = torch.load(modelPath, pickle_module=dill, map_location=torch.device('cpu'))

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save("/Users/billb/Projects/AWI/NetExploration/UNETR-nnUNetPlans_2d-DC_and_CE_loss-w-1-20-20-dill-scripted.pt") # Save

model = torch.jit.load("/Users/billb/Projects/AWI/NetExploration/UNETR-nnUNetPlans_2d-DC_and_CE_loss-w-1-20-20-dill-scripted.pt")
model.eval()

model = torch.compile(model)

# move model to device
# model = model.to(device)
# print(model)

# Add softmax to last layer
class ModelWithSoftmax(torch.nn.Module):
    def __init__(self, original_model):
        super(ModelWithSoftmax, self).__init__()
        self.original_model = original_model
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.original_model(x)
        return self.softmax(x)
    
model = ModelWithSoftmax(model)

# Save the modified model using dill
modelSoftMaxPath = "/Users/billb/Projects/AWI/NetExploration/UNETR-nnUNetPlans_2d-DC_and_CE_loss-w-1-20-20-dill-softmax.pth"
torch.save(model, modelSoftMaxPath, pickle_module=dill)

model = model.to(device)
# model = torch.compile(model)


# torch.__version__
def infer(x):
    with torch.inference_mode():
        model.eval()
        y = torch.tensor(x, dtype=torch.float32).clone().to(device).contiguous()
        output = model(y)
        del y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output.detach().cpu().numpy()

# give the model a prediction
# prediction = model(torch.tensor(zdata, dtype=torch.float32).to(device))
prediction = infer(zdata)
print(prediction)

if torch.cuda.is_available():
    import gc
    torch.cuda.empty_cache()
    gc.collect()