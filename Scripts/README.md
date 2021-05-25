* *Trilepton_read_root.py* reads ROOT files with uproot, then creates desired dataframes and variables, and saves those dataframes as .h5-files.

* *Trilepton_classifier.py* reads the .h5-files with the dataframes, and uses machine learning algorithms to do a multi-classification analysis to predict particles. The best performing classifier is saved and imported by *Trilepton_classify_Ntuples.py* later.

* *Trilepton_classify_Ntuples.py* reads original ROOT files and the dataframes made by *Trilepton_read_root.py*, and merges them together as one dataframe. Then it loads the classification model to be used for predicting leptons, which then are added to the merged dataframe. The dataframe is the saved as .csv-file and is used to convert back into ROOT and used for plotting. 

* *Trilepton_plotter.py* reads the flat dataframe, and plots manually chosen variables that are available in the dataframe.

* *environment.yml* is the environment file (Linux or Windows). Setup with anaconda:
  - Create - $ conda env create -f environment.yml
  - Activate - $ conda activate rootenv2
  - Update packages by first creating a .txt file with all present package-versions - $ pip freeze > requirements.txt (can skip this step if requirements.txt already exists). 
  - Edit *requirements.txt*, and set desired package-version or replace "==" with ">=" . Can also add other packages.
  - Upgrade the packages - $ pip install -r requirements.txt --upgrade
