# ECoG Project
## Installation 
### Anaconda 
Please download the Anaconda software package from the Anaconda **[website](https://www.anaconda.com/products/individual#windows)**.
Use the following command in the *Anaconda Prompt* to setup the environment for this repo: 
> conda env create -f environment.yml

To activate the environment use the command:
> conda activate ecog 


To update the environment file, you can export it with the command: 

> conda env export > environment.yml

#### Please make sure to remove the <i>Prefix</i>, the last row from the generated file, before you commit 

##                                                                  
## For every note in the Project tab, please make a separate branch, and issue a pull request to the master branch when the task is finished.
<br>

## Repository structure 

 - [data](#data)
	 - [raw_data](#raw_data)
	 - [preprocessed_data](#prepcessed data)
 - [preprocessing](#preprocessing)
 - [models](#models)
 - [logs](#logs)
 - [trained_models](#trained_models)
 
 <a name="data"> </a>
 #### data
 All of the data will be in this folder
 <a name="raw_data"> </a>
 #### raw_data
 The storage of the downloaded raw data is here, in a folder structure specific to the database
 <a name="preprocessing"> </a>
 #### preprocessing
 All of the different loading models are stored here and the main preprocessing methods also

 <a name="models"> </a>
 #### models
 The storage of multiple main models 
 <a name="logs"> </a>
 #### logs
 The logging goes here, but intended for offline use only 
 <a name="trained_models"> </a>
 #### trained_models
 The different trained models are stored here. The naming should be consistent with the ones in the Google Drive Excel File **models.xml**


TensorFlow GPU support is needed to run HTNet code. Install instructions can be found in the [tensorflow website](https://www.tensorflow.org/install/gpu) or in [this blog](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1).

Tested build configurations:

Linux

| NVIDIA GPU Driver | CUDA toolkit | cuDNN SDK | Python | TensorFlow |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 460.32.03 | 11.0 | 8.1.0 | 3.8.8 | 2.4.0 |
| 460.32.03 | 11.0 | ? | 3.7.10 | 2.4.1 |

Use the above configurations to avoid library conflicts.

Always check compatibility:

- [CUDA toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
- [TF](https://www.tensorflow.org/install/source#gpu)
