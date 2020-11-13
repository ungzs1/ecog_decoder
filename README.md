# ECoG Project
## Installation 
### Anaconda 
Please download the Anaconda software package from the Anaconda **[website](https://www.anaconda.com/products/individual#windows)**.
Use the following command in the *Anaconda Prompt* to setup the environment for this repo: 
> conda env create -f environment.yml

To activate the environment use the command:
> conda activate ecog 
<br>

## For every note in the Project tab, please make a separate branch, and issue a pull request to the master branch when the task is finished.
<br>

## Repository structure 

 - [data](#data)
	 - [raw_data](#raw_data)
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
