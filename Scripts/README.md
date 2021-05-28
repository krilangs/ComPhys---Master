# Classification:
* *Trilepton_read_root.py* reads ROOT files with uproot, then creates desired dataframes and variables, and saves those dataframes as .h5-files. Can also make the dataframes with the new added variables back into ROOT for plotting.

* *Trilepton_classifier.py* reads the .h5-files with the dataframes, and uses machine learning algorithms to do a multi-classification analysis to predict particles. The best performing classifier is saved and imported by *Trilepton_classify_Ntuples.py* later.

* *Trilepton_classify_Ntuples.py* reads original ROOT files and the dataframes made by *Trilepton_read_root.py*, and merges them together as one dataframe. Then it loads the classification model to be used for predicting leptons, which then are added to the merged dataframe. The dataframe is the saved as .csv-file and is used to convert back into ROOT and used for plotting. 

# Plotting:
* This folder contains the Pyton scripts used to produce the Ntuple distributions of data+MC+signal be reading ROOT-files. See the folder for description on how to run the scripts.

# 
* *environment.yml* is the environment file (Linux or Windows). *requirements.txt* contains the current package versions used in this thesis. Setup with anaconda:
  - Create - $ conda env create -f environment.yml
  - Activate - $ conda activate rootenv2
  - Update packages by first creating a .txt file with all present package-versions - $ pip freeze > requirements.txt (can skip this step if requirements.txt already exists). 
  - Edit *requirements.txt*, and set desired package-version or replace "==" with ">=" . Can also add other packages.
  - Upgrade the packages - $ pip install -r requirements.txt --upgrade

* *ROOT_env.bashrc* is the ROOT environment for using ROOT and the scripts in the Plotting folder.

* *finalized_model_150.pkl* and *finalized_model_450.pkl* are the classification models trained on the 150 GeV and 450 GeV neutrino signals.
