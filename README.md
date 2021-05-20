Some python files are a bit messy at this point.

Scripts: Contains all the scripts used in this thesis.

Plots: Contains all the plots made by the scripts used in this thesis.

* *Masteroppgave_tmp.pdf* is the master thesis (so far).

* *myfile_allevents.root* is the file with all the events (N1=50 GeV).

* *environment.yml* is the environment file (Linux or Windows). Setup with anaconda:
  - Create - $ conda env create -f environment.yml
  - Activate - $ conda activate rootenv2
  - Update packages by first creating a .txt file with all present package-versions - $ pip freeze > requirements.txt (can skip this step if requirements.txt already exists). 
  - Edit *requirements.txt*, and set desired package-version or replace "==" with ">=" . Can also add other packages.
  - Upgrade the packages - $ pip install -r requirements.txt --upgrade
  

