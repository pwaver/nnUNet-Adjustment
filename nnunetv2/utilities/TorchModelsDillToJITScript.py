# I have a bunch of PyTorch models at "/Users/billb/Projects/AWI/NetExploration" that I wish to convert to JITScript
#First, need to import the relevant packages
1+1
import dill
import torch
import torch.jit
import nnunetv2

# import torch.jit.script 

# Read the list of model names ending with "dill.pth" from the file system at directory "/Users/billb/Projects/AWI/NetExploration" into a list.
import os

directory = "/Users/billb/Projects/AWI/NetExploration"
model_files = [f for f in os.listdir(directory) if f.endswith("dill.pth")]



# Let's develop a function to convert each of these models to JITScript
model_file = model_files[0]
model = torch.load(os.path.join(directory, model_file), pickle_module=dill, map_location=torch.device('cpu'))
# Convert the model to JITScript
model_scripted = torch.jit.script(model)

# Save the JITScript model with the new file name
scripted_model_file = model_file.replace("dill.pth", "-script.pt")
model_scripted.save(os.path.join(directory, scripted_model_file))

