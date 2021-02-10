Python files are a bit messy at this point.

* *Masteroppgave_tmp.pdf* is the master thesis (so far).

* *myfile_allevents.root* is the Ntuple-file with all the events.

* *environment.yml* is the anaconda environment file (Linux or Windows):
  - Create - $ conda env create -f environment.yml
  - Activate - $ conda activate rootenv2
  - Update packages by first creating a .txt file with all present package-versions - $ pip freeze > requirements.txt (can skip this step if requirements.txt already exists). 
  - Edit *requirements.txt*, and set desired package-version or replace "==" with ">=" . Can also add other packages.
  - Upgrade the packages - $ pip install -r requirements.txt --upgrade
  
* *Trilepton_read_root.py* reads the ROOT file, uses uproot to convert to Python readable, creates desired dataframes and saves those dataframes as .h5-files.

* *Trilepton_classifier.py* reads the .h5-files with the dataframes and uses machine learning algorithms to do a multi-classification analysis to predict particles.
