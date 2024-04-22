We use here the example of a dataset named Dataset330_Angiography. It is in the nnUNet standard location of ./data/nnUNet_raw. Our data are furnished as images frames of size 512x512 pixels. The default 2d UNet planner compresses it in the encoder to 512x4x4, then decodes it with the help of skip connections back to 512x512, one for each class. On the htypothesis that 512x4x4 leads to over-regularization, we wish to reduce the compression. In the example here, we wish the encoder to go to 512x16x16. This is the meaning of the term "edge16" in the nnUNet plan.

First step is to create a new experiment planner this can inherit from the class ExperimentPlanner that is in the file default_experiment_planner.py. In this example, the inheritance class is called CompressedExperimentPlanner. The constructor has a call to super.__init__(….) as below and has within it a new reference value and a new value for min edge length.

In default_experiment_planner.py we have

class CompressedExperimentPlanner(ExperimentPlanner):
	def __init__(….):
		super.__init__(….)
		….
		self.UNet_featuremap_min_edge_length = 16
		….

Then one specifies this experiment planner class in a call to the command line program that plans the experiment (this is nnUNet terminology for designign the nnUNet structure), nnUNetv2_plan_experiment. The name of the experiment planner class is passed via a parameter -pl. The generated plan name is specified via parameter -overwrite_plans_name.

nnUNetv2_plan_experiment -d 330 -pl CompressExperimentPlanner -overwrite_plans_name plans_unet_edge16

This creates a new experiment plan file in json format. The command line program nnUNetv2_plan_experiment places the plan file in its standard place within the file structure under the data directory. 

To do this one observes the creation by the experiment planner a file under nnUNet_preprocessed/Dataset330_Angiography called plans_unet_edge16.json as specified in the call to nnUNetv2_plan_experiment above. The subsequent call to trainer then expects to find this file and furthermore find the training data in the expected location. Note that the directory where it expects to find the 2d data has "_2d" appended to the name, as plans_unet_edge16_2d, illustrated below.

To find the training in a directory plans_edge_2d

File structure:

data
	nnUNet_preprocessed
		Dataset330_Angiography
			plans_unet_edge16_2d
				Angio_0001_seq.npy (expect to copy the data here or rename its wrapper directory)
				….
			plans_unet_edge16.json (created by the call to nnUNetv2_plan_experiment)

Once this is set up one calls the training program and specifies the training configuration in a parameter -p.

There are environment variables that give this file system structure. These are given in the shell by calls as
export nnUNet_raw="/billb/github/nnUNet-Adjustment/data/nnUNet_raw"
export nnUNet_preprocessed="/billb/github/nnUNet-Adjustment/data/nnUNet_preprocessed"
export nnUNet_results="/billb/github/nnUNet-Adjustment/data/nnUNet_results"

I think these can be snuck into the conda environment. Execute conda env list to find the conda environments and their paths. Go to the environment that has nnUNet installed and look for etc/conda/activate.d. There you will create or find a file called env_vars.sh. Open it and place the export statements into it.

With the above set up, the training program command line call is something like:

nnUNetv2_train 330 2D 1 -tr nnUNetTrainer -p plans_unet_edge16

To predict, the plans file needs to be included under the parameter -p.

nnUNetv2_predict -i path-to-input-folder -o path-to-output-folder -d 330 -c 2D -tr nnUNetTrainer —disable_tta -f 1 -p plans_unet_edge16 -chk checkpoint_best.pth