Some python files are a bit messy at this point.

* *Masteroppgave_tmp.pdf* is the master thesis (so far).

* *myfile_allevents.root* is the file with all the events.

* *environment.yml* is the environment file (Linux or Windows). Setup with anaconda:
  - Create - $ conda env create -f environment.yml
  - Activate - $ conda activate rootenv2
  - Update packages by first creating a .txt file with all present package-versions - $ pip freeze > requirements.txt (can skip this step if requirements.txt already exists). 
  - Edit *requirements.txt*, and set desired package-version or replace "==" with ">=" . Can also add other packages.
  - Upgrade the packages - $ pip install -r requirements.txt --upgrade
  
* *Trilepton_read_root.py* reads the ROOT file with uproot, then creates desired dataframes and variables and saves those dataframes as .h5-files.

* *Trilepton_classifier.py* reads the .h5-files with the dataframes, and uses machine learning algorithms to do a multi-classification analysis to predict particles.

* *Trilepton_plotter.py* reads the flat dataframe, and plots manually chosen variables that are available in the dataframe.
