Based on the plotting package developed by Jesse Liu in the Electroweak compressed/Higgsino group ([HiggsinoFitter](https://gitlab.cern.ch/atlas-phys-susy-higgsino/HiggsinoFitter/tree/master/plotting)), further developed by the 2LJets group ([SusySkim2LJetsLegacy](https://gitlab.cern.ch/atlas-phys-susy-2LJetsLegacy/SusySkim2LJetsLegacy/tree/master/scripts)). This README is modified from the latter. 

# How to make plots
To plot a variable (as defined in `variables.py`) in a region (as defined in `cuts.py`), run the command

```
./plot.py -v <variable> -s <region>
```
e.g.
```
./plot.py -v met_Et -s SRhigh_0Jb
```

To list all available command line options, run

```
./plot.py -h
```


# Scripts
* `cuts.py`: defines cut selections/regions
* `variables.py`: defines the histograms
* `samples.py`: defines type (data, bkg, signal), legend and plotting color (in addition to file suffix) for the samples
* `plot.py`: the main plotting script, which imports from `cuts.py`, `variables.py` and `samples.py`, and is where you specify the paths to the samples you want to plot


# How to make the scripts run on your files

Below is a summary of lines in the scripts you need to edit in order to run on your own files.

### Configure the main code
* At the top of `plot.py`, before the `main()` function: 
   * Set the plot lables, paths to the folders where your ntuples live, the luminosity you will plot and the name of the folder where you want the plots to be saved.
* In `calc_selections()`:
  * List the background samples you want to plot in `d_samp_bkg` (corresponding to the sample names specified in `samples.py`), and the signal samples you want to plot in the `d_samp_signal` dictionaries (different signal samples plotted for different regions).

### Configure samples
* In the `configure_samples()` function in `samples.py`:
  * Specify the suffix of the data, background and signal file names you want to run on.
  * Define type, legend, color and file names in the `d_samp` dictionary.

### Configure cut selections/regions
* In the `configure_cuts()` function in `cuts.py`:
  * See which cut regions have been defined, and define your own selections/regions.

### Configure histograms
* In the `configure_vars()` function in `variables.py`:
  * Check whether histograms have already been defined for the variables you want to plot, and edit the existing or define new histograms if needed.
  * Note that a key `ntupVar` can be added to the entries in the `d_vars` dictionary, allowing the variable input when running `plot.py` to contain plotting information. This feature has been added to the 3L version of the code.


# How to make multiple plots in one go

The shell script `make_plots.sh` loops over specified lists of which variables to plot in which regions. After modifying the script with your selection of variables and  regions, run the script by executing

```
source make_plots.sh
```


# Comments

* The option to run for different periods is currently not working for this 3L version on the code. 
* The systematic uncertainty is currently set to 30% flat. To change this, define a histogram with the wanted percentage (like d_hsigZ30) at the end of the `calc_selections` function in `plot.py`, and send this new histogram into the `draw_sig_scan()` function from the `plot_selections()` function. 