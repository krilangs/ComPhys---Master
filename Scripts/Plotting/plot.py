#!/usr/bin/env python
'''
plot.py is the main script to do the plotting.
This reads the ntuples.
Makes plots of data vs MC + signals in various variables.
Configure various aspects in:
  - cuts.py
  - samples.py
  - variables.py
One specifies the samples to be plotted at the top of calc_selections() function.
'''

# So Root ignores command line inputs so we can use argparse
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(1)
from ROOT import *
import time 
from math import sqrt
from random import gauss
import os, sys, time, argparse
from array import array

from samples import *
from cuts import *
from variables import *


# Labels

ntuple_version = 'v2.3c'
ATL_status = 'Internal'
NTUP_status = 'EWK3L ' + ntuple_version

# Path to directory of full ntuples
#TOPPATH = '/scratch2/Master_krilangs/SUSY_LOOSE_NOMINAL'

#TOPPATH = '/scratch2/Master_krilangs/Trilepton_Ntuples/Skimslim'  # Original Ntuples
TOPPATH = '/scratch2/Master_krilangs/Trilepton_Ntuples/Ntuples_class_ROOT'  # Classified Ntuples


# Paths to subdirectories                                                       
DATAPATH = TOPPATH
BKGPATH = TOPPATH
SIGPATH = TOPPATH
DATAPATH1516 = TOPPATH + '/data1516_mc16a'
DATAPATH17 = TOPPATH + '/data17_mc16d'
DATAPATH18 = TOPPATH + '/data18_mc16e'
BKGPATH_MC16a = TOPPATH + '/data1516_mc16a'
BKGPATH_MC16d = TOPPATH + '/data17_mc16d'
BKGPATH_MC16e = TOPPATH + '/data18_mc16e'
# SIGPATH_MC16a = TOPPATH + 'SUSY2_Signal_mc16a/'
# SIGPATH_MC16cd = TOPPATH + 'SUSY2_Signal_mc16cd/'
# SIGPATH_MC16e = TOPPATH + 'SUSY2_Signal_mc16e/'

# Luminosity 
lumi_15_18 = 139 # 3L analysis
lumi_15_16 = 36.2 # [1/fb] 15+16
lumi_17 = 44.3 # [1/fb] 17
lumi_18 = 59.9 # [1/fb] 18
# lumi_15_17 = lumi_15_16 + lumi_17 # [1/fb] 15+16+17
# lumi_15_18 = lumi_15_16 + lumi_17 + lumi_18 # [1/fb] 15+16+17+18 (140.4)

# The path/folder name where the plots will be saved -- the folder will be created if it doesn't exist already                                                                                        
#savedir = '../Trilepton_Plots/ROOT_plots/'  # 3L_original_Ntuples
#savedir = '../Trilepton_Plots/Feature_plots/' # New feature plots
savedir = '../Trilepton_Plots/Class_ROOT_plots/Benchmark'  # 3L_classified_Ntuples



savedir_15_16 = savedir + 'data15-16_vs_mc16a'
savedir_17 = savedir + 'data17_vs_mc16d'
savedir_18 = savedir + 'data18_vs_mc16e_SR'
# savedir_15_17 = savedir + 'data15-17_vs_mc16ad'
#savedir_15_18 = savedir + 'data15-18_vs_mc16ade'

# Path to folder where yields will be saved
yieldsdir = './yields_plots/'

use_mc_fakes = True 
useLooseNominal = False

# Text size as percentage
text_size = 0.045

#____________________________________________________________________________
def main():
  
  t0 = time.time()
  
  global DATAPATH, BKGPATH, SIGPATH, data_period, sig_reg, isData15_16, isData17, isData18, isData15_17, isData15_18, savedir, lumi, allNCuts, Signal

  #================================================
  # default values
  var     = 'met_Et'
  # var = 'L3_maxMll_0_200_lower'
  sig_reg = 'SRhigh_0Jb'
  data_period = 'data18'
  save_label = 'non'
  isData15_16 = False
  isData17 = False
  isData18 = False
  isData15_17 = False
  isData15_18 = True
  savedir = savedir#_15_18
  lumi    = lumi_15_18
  BKGPATH = BKGPATH 
  SIGPATH = SIGPATH 
  DATAPATH = DATAPATH
  # BKGPATH = BKGPATH_MC16a
  # SIGPATH = SIGPATH_MC16a  
  ttbarSamp = 'ttbar'
  unblind = False
  cutArrow = False
  IsLogY = True
  showOverflow = True
  allNCuts = False
  TriSig = False 
  TriLep = False
  Signal = "150"

  # Check if user has inputted variables or not
  parser = argparse.ArgumentParser(description='Analyse background/signal TTrees and make plots.')
  parser.add_argument('-v', '--variable',  type=str, nargs='?', help='String name for the variable to make N-1 in. Either as appearing in TTree, or, if added, with additional plot information', default=var) # Option for the var "name" to include plotting information has been added for 3L analysis
  # parser.add_argument('-v', '--variable',  type=str, nargs='?', help='String name for the variable (as appearing in the TTree) to make N-1 in.', default=var)
  parser.add_argument('-s', '--sigReg',    type=str, nargs='?', help='String name of selection (signal/control) region to perform N-1 selection.', default=sig_reg)
  parser.add_argument('-l', '--lumi',      type=str, nargs='?', help='Float of integrated luminosity to normalise MC to.', default=lumi)
  parser.add_argument('-t', '--ttbarSamp', type=str, nargs='?', help='ttbar sample to use.', default=ttbarSamp)
  parser.add_argument('-u', '--unblind',   type=str, nargs='?', help='Should the SRs be unblinded?')
  parser.add_argument('-p', '--period',    type=str, nargs='?', help='Set data period: data15-16, data17, data18 or data15-18.')
  parser.add_argument('-L', '--label',     type=str, nargs='?', help='Append descriptive label to output filename.', default=save_label)
  parser.add_argument('-a', '--cutArrow',  action='store_true', help='Draw arrows where cuts are placed for N-1 plots.')
  parser.add_argument('-n', '--noLogY',    action='store_true', help='Do not draw log Y axis.')
  parser.add_argument('-o', '--notShowOverflow',  action='store_true', help='Do not include overflow in bin N.')
  parser.add_argument('-N', '--allNCuts',  action='store_true', help='Keep cut on variable to be plotted (if it exists).')
  parser.add_argument('-T', '--TriSig', action='store_true', help='Add desired 3L signals on top of plots.')
  parser.add_argument('-Tri', '--TriLep', action='store_true', help='Add desired trilepton signals on top of plots.')
  parser.add_argument('-S', '--Signal', type =str, nargs='?', help='Add desired signal, 150 or 450')

  args = parser.parse_args()
  if args.variable:
    var = args.variable
  if args.sigReg:
    sig_reg = args.sigReg
  if args.lumi:
    lumi = args.lumi
  if args.label:
    save_label = args.label
  if args.ttbarSamp:
    ttbarSamp = args.ttbarSamp
  if args.Signal:
    Signal = args.Signal
  if args.period == 'data15-16':
    data_period = args.period
    isData15_18 = False
    isData15_16 = True
    BKGPATH = BKGPATH_MC16a 
    DATAPATH = DATAPATH1516
    savedir = savedir_15_16
    lumi = lumi_15_16
  elif args.period == 'data17':
    data_period = args.period
    isData15_18 = False
    isData17 = True
    BKGPATH = BKGPATH_MC16d 
    DATAPATH = DATAPATH17
    savedir = savedir_17
    lumi = lumi_17
  elif args.period == 'data18':
    data_period = args.period
    isData15_18 = False
    isData18 = True
    BKGPATH = BKGPATH#_MC16e
    DATAPATH = DATAPATH#18
    savedir = savedir_18
    lumi = lumi_18
  elif args.period == 'data15-17':
    data_period = args.period
    isData15_18 = False
    isData15_17 = True
    savedir = savedir_15_17
    lumi = lumi_15_17
  elif args.period == 'data15-18':
    data_period = args.period
    isData15_18 = True
    savedir = savedir_15_18
    lumi = lumi_15_18
    
  # I know we could just use a bool argument here, but maybe safer to 
  # require the string, so it's harder to unblind by mistake!
  if args.unblind == 'True':
    unblind = True
  if args.cutArrow:
    cutArrow = True
  if args.noLogY:
    IsLogY = False
  if args.notShowOverflow:
    showOverflow = False
  if args.allNCuts:
    allNCuts = True
  if args.TriSig:
    TriSig = True
  if args.TriLep:
    TriLep = True

  print( '\n=========================================' )
  print( 'Data directory: {0}'.format(DATAPATH) )
  print( 'MC directory: {0}'.format(BKGPATH) ) 
  print( 'Signal directory: {0}'.format(SIGPATH) )
  # if isData15_16:
  #   print( 'MC directory: {0}'.format(BKGPATH_MC16a) )
  # elif isData17:

  # elif isData18:
  #   print( 'MC directory: {0}'.format(BKGPATH_MC16e) )
  # elif isData15_18:
  #   print( 'MC directory: {0}'.format(BKGPATH_MC16a) )
  #   print( 'MC directory: {0}'.format(BKGPATH_MC16cd) )
  #   print( 'MC directory: {0}'.format(BKGPATH_MC16e) )
  print( 'Plotting input: {0}'.format(var) ) # 3L analysis
  # print( 'Plotting variable: {0}'.format(var) )
  print( 'Selection region: {0}'.format(sig_reg) )
  print( 'Data-taking period: {0}'.format(data_period) )
  print( 'Normalising luminosity: {0}'.format(lumi) )
  # print( 'ttbar Sample: {0}'.format(ttbarSamp) )                                                                                                  
  print( '=========================================\n' )
  
  #================================================
  # Make (relative) save directory if needed 
  mkdir(savedir)
 
  save_var = var
  # Convert maths characters to legit file names
  if '/' in var:
    save_var = save_var.replace('/', 'Over', 1) 
  if '(' in var:
    save_var = save_var.replace('(', '', 1)
  if ')' in var:
    save_var = save_var.replace(')', '', 1)
  if IsLogY:
    save_name = savedir + '/hist1d_{0}_{1}_{2}'.format(save_var, sig_reg, Signal)   # _{2}  (Signal)
  if not IsLogY:
    save_name = savedir + '/hist1d_{0}_{1}_noLogY'.format(save_var, sig_reg)
  if save_label is not 'non':
    save_name = save_name + '_' + save_label
    
  # If plotting single data period when running over merged data15-18 ntuples
  # select run numbers to run on according to data period
  if isData15_16:
    add_cut = 'RandomRunNumber < 320000'
  elif isData17:
    add_cut = '(RandomRunNumber > 320000 && RandomRunNumber < 348000)'
  elif isData18:
    add_cut = 'RandomRunNumber > 348000'
  else:
    add_cut = '1'

  annotate_text = ''
  if isData15_16:
    annotate_text = '2015-16 data vs. mc16a'
  elif isData17:
    annotate_text = '2017 data vs. mc16d'
  elif isData18:
    annotate_text = '2018 data vs. mc16e'
  elif isData15_18:
     annotate_text = '2015-18 data vs. mc16a+d+e'  

  #==========================================================
  # List samples to analyse 
  #==========================================================
  if use_mc_fakes:
    ttbarSamp = 'ttbar' #'alt_ttbar_nonallhad'
  else: 
    ttbarSamp = 'ttbar' 

  calc_selections(var, add_cut, lumi, save_name, sig_reg, Signal, annotate_text, ttbarSamp, unblind, cutArrow, IsLogY, showOverflow, allNCuts, TriSig, TriLep)
  
  tfinish = time.time()
  telapse = tfinish - t0
  m, s = divmod(telapse, 60)  
  print('{} min {} s'.format(int(m), int(round(s))))

#____________________________________________________________________________
def calc_selections(var, add_cuts, lumi, save_name, sig_reg, Signal, annotate_text='', ttbarSamp='ttbar', unblind=False, cutArrow=False, IsLogY=True, showOverflow=True, allNCuts=False, TriSig=False, TriLep=False):
  '''
  Extract trees given a relevant variable
  '''
  #==========================================================
  # Prepare information and objects for analysis and plots
  #==========================================================

  ### All lists converted to dicts for 3L analysis
  
  d_samp_bkg = {
   'diboson2L' : ['diboson2L'],
   'singletop' : ['singletop'],
   #'fakes'     : ['fakes'],
   'ttbar'     : ['ttbar'],
   'ttbar+X'   : ['topOther','higgs'],
   'triboson'  : ['triboson'],
   'diboson4L' : ['diboson4L'],
   'diboson3L' : ['diboson3L'],
   'Zjets'     : ['Zjets'],
   }

  d_samp_signal_A_B = {
    'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_150_140_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_140_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3'] 
  }

  d_samp_signal_C = {
    'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3'] 
  }

  d_samp_signal_D = {
    'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3'] 
  }

  d_samp_signal_E = {
    'MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3'] 
  }

  d_samp_signal_F = {
    'MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3'  : ['MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3'  : ['MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3'] 
  }

  d_samp_signal_G = {
    'MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3'],   
    'MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3']    
  }

  d_samp_signal_3L = {
    'MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3'],
    'MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3']
  }

  d_samp_signal_TriLep = {
    'LFCMN1150' : ['LFCMN1150'],
    'LFCMN1450' : ['LFCMN1450']
}

  if 'SRhigh_nJa' in sig_reg or 'SRlow_0Jb' in sig_reg or 'SRlow_nJb' in sig_reg or 'SRhigh_0Jb' in sig_reg or 'SRhigh_nJb' in sig_reg:
    d_samp_signal = d_samp_signal_A_B
  elif 'SRlow_0Jc' in sig_reg or 'SRlow_nJc' in sig_reg or 'SRhigh_0Jc' in sig_reg or 'SRhigh_nJc' in sig_reg:
    d_samp_signal = d_samp_signal_C
  elif 'SRlow_0Jd' in sig_reg or 'SRlow_nJd' in sig_reg or 'SRhigh_0Jd' in sig_reg or 'SRhigh_nJd' in sig_reg:
    d_samp_signal = d_samp_signal_D
  elif 'SRlow_0Je' in sig_reg or 'SRlow_nJe' in sig_reg or 'SRhigh_0Je' in sig_reg or 'SRhigh_nJe' in sig_reg:
    d_samp_signal = d_samp_signal_E
  elif 'SRlow_0Jf1' in sig_reg or 'SRlow_0Jf2' in sig_reg or 'SRlow_nJf1' in sig_reg or 'SRlow_nJf2' in sig_reg or 'SRhigh_0Jf1' in sig_reg or 'SRhigh_0Jf2' in sig_reg or 'SRhigh_nJf' in sig_reg:
    d_samp_signal = d_samp_signal_F
  elif 'SRlow_0Jg1' in sig_reg or 'SRlow_0Jg2' in sig_reg or 'SRlow_nJg1' in sig_reg or 'SRlow_nJg2' in sig_reg or 'SRhigh_0Jg1' in sig_reg or 'SRhigh_0Jg2' in sig_reg or 'SRhigh_nJg' in sig_reg:
    d_samp_signal = d_samp_signal_G
  elif TriSig == True:
    d_samp_signal = d_samp_signal_3L
  elif TriLep == True:
    d_samp_signal = d_samp_signal_TriLep
  else: # CRs and VRs
    #d_samp_signal = d_samp_signal_A_B
    d_samp_signal = {}

  d_samp_other = d_samp_signal

  # Blind the SRs
  if ('SR' not in sig_reg) or unblind or ('ISR' in sig_reg and unblind) : # last OR: workaround to be able to plot data in non-SR ISR regions
    d_data = {'data' : ['data']}
    d_samp_other = d_samp_other.copy(); d_samp_other.update(d_data) # corresponds to l_samp_other = ['data'] + l_samp_other

  # Total dict of samples
  d_samps = d_samp_bkg.copy(); d_samps.update(d_samp_other) # corresponds to l_samp = l_samp_bkg + l_samp_other 
  
  # Get dictionary of histogram configurations
  d_vars = configure_vars(sig_reg)

  # Get variable to be plotted (feature added for the 3L analysis)
  if 'ntupVar' in d_vars[var]: 
    var_plot = d_vars[var]['ntupVar'] 
  else:
    var_plot = var

  # Obtain the number of bins with their xmin and xmax
  hNbins = d_vars[var]['hXNbins']
  hXmin  = d_vars[var]['hXmin'] 
  hXmax  = d_vars[var]['hXmax']  

  # Check variable_bin
  variable_bin = False
  hXarray = []
  if hNbins == 'var':
    variable_bin = True
    hXarray = d_vars[var]['binsLowE'] 
    hNbins = len(hXarray) - 1

  # Obtain cut to apply (string)
  normCutsAfter = configure_cuts(var_plot, add_cuts, sig_reg, isData18, allNCuts)
  
  # Get dictionary defining sample properties
  d_samp = configure_samples(isData15_16, isData17, isData18, isData15_18, Signal)  
  
  # Declare stacked background  
  hs = THStack('','')
  hs_intgl_low = THStack('','') # lower cut integral (for significance cut)
  hs_intgl_upp = THStack('','') # upper cut integral (for significance cut)
  # hs_intgl_overflow = THStack('','') # overflow included in the bin N
 
  # Initialise objects to fill in loop 
  d_files = {}
  d_hists = {}
  d_yield = {}
  d_yieldErr = {}
  # d_yield_overflow = {}
  nTotBkg = 0 # yield of background
  nVarBkg = 0 # variance of background
  # nTotBkg_overflow = 0 # yield of background with overflow included in bin N
  # nVarBkg_overflow = 0 # variance of background with overflow included in bin N 
  
  # l_bkg = [] 
  l_sig = []

  h_dat = 0
  N_dat = 0
  # N_dat_overflow = 0

  #==========================================================
  # loop through samples, fill histograms
  #==========================================================
  l_styles = [7] * len(d_samp_signal)
  Nsignal_count = 0
  # Nsignal_overflow_count = 0

  ### Loop over dicts, 3L analysis

  # First loop over groups, not sorted after yields
  for group,samples in d_samps.iteritems():
    # Make sure not to unblind by mistake
    if 'SR' in sig_reg and 'data' in group and not unblind:
      continue

    # Prepare dictionary for current group sample paths
    d_files[group] = [] 

    # Loop over samples in group
    for i,samp in enumerate(samples):
      # Obtain sample attributes 
      sample_type = d_samp[samp]['type']
      path        = d_samp[samp]['path']
  
      # Choose full path of sample by its type  
      full_path = ''
      #full_path_data15_16 = ''
      #full_path_data17 = ''
      #full_path_data18 = ''

      if sample_type == 'sig':        
        full_path = SIGPATH + '/' + path
        # full_path_data15_16 = SIGPATH_MC16a + '/' + path
        # full_path_data17    = SIGPATH_MC16cd + '/' + path
        # full_path_data18    = SIGPATH_MC16e + '/' + path
        l_sig.append(samp)
    
      elif sample_type == 'bkg':
        full_path = BKGPATH + '/' + path
        # full_path_data15_16 = BKGPATH_MC16a + '/' + path
        # full_path_data17    = BKGPATH_MC16cd + '/' + path
        # full_path_data18    = BKGPATH_MC16e + '/' + path
        # l_bkg.append(samp) 
                
      elif sample_type == 'data':
        full_path = DATAPATH + '/' + path
        # full_path_data15_16 = DATAPATH + '/' + 'data15-16' + path
        # full_path_data17 = DATAPATH + '/' + 'data17' + path
        # full_path_data18 = DATAPATH + '/' + 'data18' + path    

      cutsAfter = normCutsAfter 

      # Store full paths in dictonary 
      d_files[group] += [full_path]
      # d_files[group] += [full_path_data15_16, full_path_data17, full_path_data18]
      
    # Obtain histogram from file and store to dictionary entry
    d_hists[group] = tree_get_th1f( group, samples, d_files[group], var_plot, cutsAfter, hNbins, hXmin, hXmax, lumi, variable_bin, hXarray, showOverflow )
    
  # Second loop over groups, sorted after yields to get correct stack order
  # NB! 'samp' used for 'group' hereafter, so syntax can stay unchanged 
  for samp,d_hists_output in sorted(d_hists.items(), key = lambda item : (item[1][1])):
    # Make sure not to unblind by mistake
    if 'SR' in sig_reg and 'data' in samp and not unblind:
      continue

    # ---------------------------------------------------- 
    # Stacked histogram: construct and format
    # ---------------------------------------------------- 
    # Extract key outputs of histogram 
    hist        = d_hists[samp][0]
    nYield      = d_hists[samp][1]
    h_intgl_low = d_hists[samp][2]
    h_intgl_upp = d_hists[samp][3]
    nYieldErr   = d_hists[samp][4]
    # h_intgl_overflow   = d_hists[samp][5]
    # nYield_overflow    = d_hists[samp][6]
    # nYieldErr_overflow = d_hists[samp][7]
        
    # Fill dictionary, samp : nYield, number of events for each sample
    d_yield[samp]    = nYield
    d_yieldErr[samp] = nYieldErr
    # d_yield_overflow[samp]= nYield_overflow
  
    sample_type = d_samp[samp]['type']
  
    # Add background to stacked histograms
    if sample_type == 'bkg':

      # Placed here in order to get black contours for the individual bkg hists
      f_color = d_samp[samp]['f_color'] 
      format_hist(hist, 1, kBlack, 1, f_color, 1001, 0) 
      
      hs.Add(hist)
      hs_intgl_low.Add(h_intgl_low)
      hs_intgl_upp.Add(h_intgl_upp)
      # hs_intgl_overflow.Add(h_intgl_overflow)

      if nYield > 0: 
        nTotBkg  += nYield
      nVarBkg  += nYieldErr ** 2
      # nTotBkg_overflow  += nYield_overflow
      # nVarBkg_overflow  += nYieldErr_overflow ** 2
      
    if sample_type == 'sig':
      l_color = d_samp[samp]['l_color']
      # format_hist(hist, 3, l_color, 7, f_color=0)
      format_hist(hist, 3, l_color, l_styles[Nsignal_count], f_color=0)
      Nsignal_count += 1
    
    if sample_type == 'data':
      h_dat = hist
      format_hist(h_dat,  1, kBlack, 1)
      N_dat = nYield
      # N_dat_overflow = nYield_overflow

  errStatBkg = sqrt( nVarBkg ) # Treat total statistical error as sum in quadrature of sample stat errors
  errTotBkg  = errStatBkg
  # errTotBkg  = sqrt( errStatBkg**2 + (0.2 * nTotBkg) ** 2 )
  
  print('errStatBkg: {0:.3f}, sqrtB: {1:.3f}, errTotBkg: {2:.3f}'.format(errStatBkg, sqrt(nTotBkg), errTotBkg))
  print('==============================================')
  print('{0}, Data, {1}'.format(sig_reg, N_dat))
  print('----------------------------------------------')
  # print('{0}, Total bkg, {1:.3f} +/- {2:.3f}'.format(sig_reg, nTotBkg, errTotBkg))
  print('{0}, Total bkg, {1:.1f}, {2:.1f}'.format(sig_reg, nTotBkg, errTotBkg))
  print('----------------------------------------------')

  # Create and open to store yields in
  if allNCuts:
    file_path = yieldsdir+sig_reg+'_yields.dat'
    if os.path.exists(file_path):
      os.remove(file_path)
    yields_file = open(file_path, 'a')
    yields_file.write('samp \t yield \t yieldErr \n')

  # Legend for signals, data and total bkg yield 
  # First, for l_samp_other = (['data'] +) l_samp_signal
  leg = mk_leg(0.44, 0.68, 0.65, 0.93, sig_reg, d_samp_other, d_samp, nTotBkg, d_hists, d_yield, d_yieldErr, yields_file, sampSet_type='bkg', txt_size=0.033) # Also writes yields to above file
  # Then, legend with breakdown of background by sample
  d_bkg_leg = {}
  l_bkg_leg = ['samp_bkg']
  d_bkg_leg['samp_bkg'] = mk_leg(0.69, 0.60, 0.90, 0.93, sig_reg, d_samp_bkg, d_samp, nTotBkg, d_hists, d_yield, d_yieldErr, yields_file, sampSet_type='bkg', txt_size=0.033) 
   
  # Close yields file 
  if allNCuts:
    yields_file.close()

  ('==============================================')

  #============================================================
  # Make MC error histogram (background uncertainty hatching)
  pc_sys = 0 # Percentage systematic uncertainty
  h_mcErr = mk_mcErr(hs, pc_sys, hNbins, hXmin, hXmax, variable_bin, hXarray)
  h_mcErr.SetLineWidth(2)
  h_mcErr.SetLineColorAlpha(kRed+1, 1.0)
  h_mcErr.SetFillStyle(3254) # Hatching 
  h_mcErr.SetFillColor(kGray+1)
  # h_mcErr.SetMarkerSize(0)
  # h_mcErr.SetMarkerColorAlpha(kWhite, 0)
  if 'Pass' in sig_reg or 'presel' in sig_reg:
    leg.AddEntry(h_mcErr, 'SM total ({0:.1f})'.format(nTotBkg), 'lf')
    # leg.AddEntry(h_mcErr, 'SM stat #oplus 20% syst ({0:.1f})'.format(nTotBkg), 'lf')
  else:
    leg.AddEntry(h_mcErr, 'SM total ({0:.1f})'.format(nTotBkg), 'lf') 
    # leg.AddEntry(h_mcErr, '#scale[0.8]{SM total} ' + '({0:.1f})'.format(nTotBkg), 'lf')
    # leg.AddEntry(h_mcErr, '#scale[0.8]{SM stat #oplus 20% syst} ' + '({0:.1f} #pm {1:.1f})'.format(nTotBkg, errTotBkg), 'lf')    
  # leg.AddEntry(h_mcErr, 'SM 20% syst ({0:.1f})'.format(nTotBkg), 'lf')
  
  #============================================================
  # Now all background histogram and signals obtained
  # Proceed to make significance scan
  #============================================================
  # First evaluate the the signal to background ratio
  # pc_sigOverBkg = 100 * ( d_yield[signal_samp] / float( nTotBkg ) )
  # leg.AddEntry(0, 'Sig / Bkg = {0:.1f}%'.format(pc_sigOverBkg), '')
 
  # Dicitonary for histograms and its significance plots
  # In format {samp_name : histogram}
  d_hsig = {}
  d_hsigZ30 = {}
  d_hsigZ05 = {}
  # d_hsig_overflow = {}
  
  # Obtain direction of cut 
  cut_dir = d_vars[var]['cut_dir'] 
  
  # Calculate significances for signals only
  for samp,d_hists_output in sorted(d_hists.items(), key = lambda item : (item[1][1])): # 3L analysis
  # for samp in l_samp:
    sample_type = d_samp[samp]['type']
    if sample_type == 'sig':
      d_hsig[samp] = d_hists[samp][0]
      h_signal_low = d_hists[samp][2]
      h_signal_upp = d_hists[samp][3]
      # d_hsig_overflow[samp] = d_hists[samp][5]
      if 'SR' in sig_reg and not unblind :
        # Significance based on cutting to left (veto right)
        if 'upper' in cut_dir: # 3L analysis
        # if 'left' in cut_dir:
          d_hsigZ30[samp] = mk_sigZ_plot(h_signal_low, hs_intgl_low, 30, hNbins, hXmin, hXmax)
          d_hsigZ05[samp] = mk_sigZ_plot(h_signal_low, hs_intgl_low, 5, hNbins, hXmin, hXmax)
        # Significance based on cutting to right (veto left)
        if 'lower' in cut_dir: # 3L analysis
        # if 'right' in cut_dir:
          d_hsigZ30[samp] = mk_sigZ_plot(h_signal_upp, hs_intgl_upp, 30, hNbins, hXmin, hXmax)
          d_hsigZ05[samp] = mk_sigZ_plot(h_signal_upp, hs_intgl_upp, 5, hNbins, hXmin, hXmax)
  '''
  if 'SR' not in sig_reg:  
    # ensure to manually set the bin error so ratio plot is correct
    for mybin in range( h_dat.GetXaxis().GetNbins() + 1 ):  
      yval = h_dat.GetBinContent(mybin)
      yerr = h_dat.GetBinError(mybin)
      h_dat.SetBinError(mybin, yerr)
  '''
  #============================================================
  # proceed to plot
  #if showOverflow:
  #  plot_selections(var, hs_overflow, d_hsig_overflow, h_dat_overflow, h_mcErr, d_, d_hsigZ30, leg, l_bkg_leg, d_bkg_leg, lumi, save_name, pc_sys, sig_reg, nTotBkg, l_sig, cutsAfter, annotate_text, variable_bin, unblind, cutArrow, IsLogY)
  #else:
  plot_selections(var, hs, d_hsig, h_dat, h_mcErr, d_hsigZ05, d_hsigZ30, leg, l_bkg_leg, d_bkg_leg, lumi, save_name, pc_sys, sig_reg, nTotBkg, l_sig, cutsAfter, annotate_text, variable_bin, unblind, cutArrow, IsLogY)
 
  return nTotBkg

#____________________________________________________________________________
def plot_selections(var, h_bkg, d_hsig, h_dat, h_mcErr, d_hsigZ05, d_hsigZ30, leg, l_bkg_leg, d_bkg_leg, lumi, save_name, pc_sys, sig_reg, nTotBkg, l_sig, cutsAfter, annotate_text, variable_bin, unblind=False, cutArrow=False, IsLogY=True):
  '''
  plots the variable var given input THStack h_bkg, one signal histogram and legend built
  makes a dat / bkg panel in lower part of figure
  to-do: should be able to read in a list of signals
  '''
  print('Proceeding to plot')
  
  # gPad left/right margins
  gpLeft = 0.17
  gpRight = 0.05
  
  d_vars = configure_vars(sig_reg)
 
  #==========================================================
  # Build canvas
  can  = TCanvas('','',1000,1000)
  customise_gPad()
  
  pad1 = TPad('pad1', '', 0.0, 0.40, 1.0, 1.0)
  pad2 = TPad('pad2', '', 0.0, 0.00, 1.0, 0.4)
  pad1.Draw()
  pad1.cd()
  
  if IsLogY:
    pad1.SetLogy()
  customise_gPad(top=0.03, bot=0.04, left=gpLeft, right=gpRight)
  # customise_gPad(top=0.03, bot=0.20, left=gpLeft, right=gpRight)

  #=============================================================
  # Draw and decorate

  # Draw bkg
  h_bkg.Draw('hist')

  # Clone the total background histogram to draw the line
  h_mcErr_clone = h_mcErr.Clone()
  h_mcErr_clone.SetFillColorAlpha(kWhite, 0)
  h_mcErr_clone.SetFillStyle(0)
  h_mcErr.Draw('same e2')
  h_mcErr_clone.Draw('same hist')

  # Draw signal samples
  # Placed here so that the signal lines comes on top of the total bkg line
  for samp in d_hsig:
    print('Drawing {0}'.format(samp))
    d_hsig[samp].Draw('hist same') #e2 = error coloured band
    
  # Draw data  
  # IMPORTANT: only draw data for control regions until unblinded
  if 'SR' not in sig_reg or unblind:
    #pass
    h_dat.Draw('hist same ep')
    # Data point size 
    h_dat.SetMarkerSize(1.4)
    h_dat.SetLineWidth(2)
  
  leg.Draw('same')
  for bkg_leg in l_bkg_leg:
    d_bkg_leg[bkg_leg].Draw('same')
 
  ''' 
  # Put a white outline around the data points
  h_dat_outline = h_dat.Clone()
  h_dat_outline.SetMarkerStyle(25) # 25 = open square
  h_dat_outline.SetMarkerColor(kWhite)
  h_dat_outline.Draw('hist same ep')
  '''
  
  #==========================================================
  # Calculate bin width
  
  hNbins = d_vars[var]['hXNbins']
  hXmin  = d_vars[var]['hXmin']
  hXmax  = d_vars[var]['hXmax']
  if not variable_bin:
    binWidth = (hXmax - hXmin) / float(hNbins)
  
  # Label axes of top pad
  xtitle = ''
  binUnits = d_vars[var]['units']
  if variable_bin:
    ytitle = 'Events / bin'
  elif 0.1 < binWidth < 1:
    ytitle = 'Events / {0:.2f} {1}'.format(binWidth, binUnits)
  elif binWidth <= 0.1:
    ytitle = 'Events / {0:.2f} {1}'.format(binWidth, binUnits)
  elif binWidth >= 1:
    ytitle = 'Events / {0:.0f} {1}'.format(binWidth, binUnits)
  enlargeYaxis = True
  # if 'Pass' in sig_reg or 'preselect':
  if 'Pass' in sig_reg or 'preselect' in sig_reg:
    enlargeYaxis = True
  
  customise_axes(h_bkg, xtitle, ytitle, 2.8, IsLogY, enlargeYaxis)
  
  #==========================================================
  # Arrow to indicate where cut is
  # Case 2-sided cuts
  
  # Set height of arrow
  ymin_Ar = gPad.GetUymin()
  ymax_Ar = h_bkg.GetMaximum()
  if IsLogY:
    ymax_Ar = 80
  if not IsLogY:
    ymax_Ar = 0.8*ymax_Ar
  # Arrow width is 5% of the maximum x-axis bin 
  arr_width = hXmax * 0.06
  if 'cut_pos2' in d_vars[var].keys():
    cut_pos2 = d_vars[var]['cut_pos2']
    cut_dir2 = d_vars[var]['cut_dir2']
    cutAr2   = cut_arrow(cut_pos2, ymin_Ar, cut_pos2, ymax_Ar, cut_dir2, 0.012, arr_width)
    if cutArrow:
      cutAr2[0].Draw()
      cutAr2[1].Draw()
  # Otherwise 1-sided cut 
  cut_pos = d_vars[var]['cut_pos']
  cut_dir = d_vars[var]['cut_dir']
  cutAr = cut_arrow(cut_pos, ymin_Ar, cut_pos, ymax_Ar, cut_dir, 0.012, arr_width)
  if cutArrow:
    cutAr[0].Draw()
    cutAr[1].Draw()

  # Replace -mm with mu mu
  if '-ee' in sig_reg:
    sig_reg = sig_reg.replace('-ee', ' ee', 1)
  if '-mm' in sig_reg:
    sig_reg = sig_reg.replace('-mm', ' #mu#mu', 1)
  if '-em' in sig_reg:
    sig_reg = sig_reg.replace('-em', ' e#mu', 1)
  if '-me' in sig_reg:
    sig_reg = sig_reg.replace('-me', ' #mue', 1)
  if '-ee-me' in sig_reg:
    sig_reg = sig_reg.replace('-ee-me', ' ee+#mue', 1)
  if '-mm-em' in sig_reg:
    sig_reg = sig_reg.replace('-mm-em', ' #mu#mu+e#mu', 1)

  if '-SF' in sig_reg:
    sig_reg = sig_reg.replace('-SF', ' ee+#mu#mu', 1)
  if '-DF' in sig_reg:
    sig_reg = sig_reg.replace('-DF', ' e#mu+#mue', 1)
  if '-AF' in sig_reg:
    sig_reg = sig_reg.replace('-AF', ' ee+#mu#mu+e#mu+#mue', 1)
  
  #==========================================================
  # Text for ATLAS, energy, lumi, region, ntuple status
  myText(0.22, 0.87, '#bf{#it{ATLAS}} ' + ATL_status, text_size*1.2, kBlack)
  myText(0.22, 0.81, '13 TeV, {0:.1f}'.format(float(lumi)) + ' fb^{#minus1}', text_size*1.1, kBlack) 
  myText(0.22, 0.77, sig_reg, text_size*0.7, kBlack) 
  myText(0.22, 0.73, NTUP_status, text_size*0.7, kGray+1) 

  if not annotate_text == '':
    myText(0.22, 0.69, annotate_text, text_size*0.7, kGray+1) 
  
  gPad.RedrawAxis() 
  
  #==========================================================
  # go to pad 2: significance panel
  #==========================================================
  can.cd()
  pad2.Draw()
  pad2.cd()
  customise_gPad(top=0.05, bot=0.39, left=gpLeft, right=gpRight)
  
  varTeX = 'tlatex'
  
  Xunits = d_vars[var]['units']
  if Xunits == '':
    #xtitle = '{0}'.format( d_vars[var]['tlatex'])
    xtitle = '{0}'.format( d_vars[var][varTeX])
  else:
    xtitle = '{0} [{1}]'.format( d_vars[var][varTeX], Xunits ) 
  
  # SRs draw significance scans, CRs draw  
  if 'SR' in sig_reg and not unblind:
    draw_sig_scan(l_sig, d_hsigZ30, cut_dir, xtitle, hXmin, hXmax) #d_hsigZ30 = 30% uncertainty used!
    gPad.RedrawAxis() 

    # Add gray line at zero for reference, 3L analysis
    gPad.Update()
    uxmin, uxmax = gPad.GetUxmin(), gPad.GetUxmax()
    lzero = TLine(uxmin,0,uxmax,0)
    lzero.SetLineColor(kGray+1)
    lzero.SetLineWidth(1)
    lzero.SetLineStyle(2)
    lzero.Draw()
    
  if 'SR' not in sig_reg or unblind:
    #==========================================================
    # MC error ratio with MC
    h_mcErrRatio = h_mcErr.Clone()
    h_mcErrRatio.Divide(h_bkg.GetStack().Last())
    h_mcErrRatio.SetFillStyle(3245)
    h_mcErrRatio.SetFillColor(kGray+2)
    h_mcErrRatio.SetMarkerSize(0) 
    h_mcErrRatio.Draw('e2')
    
    # Draw line for the ratios
    l = draw_line(hXmin, 1, hXmax, 1, color=kGray+2, style=1) 
    l.Draw()
    
    # Draw data on top
    hRatio = h_dat.Clone()
    hRatio.Divide(h_bkg.GetStack().Last())  
    hRatio.Draw('same ep') 
    
    # Ensure uncertainties in ratio panel are consistent with upper plot
    for ibin in xrange(0, hRatio.GetNbinsX()+1) :
      ratioContent = hRatio.GetBinContent(ibin)
      dataError = h_dat.GetBinError(ibin)
      dataContent = h_dat.GetBinContent(ibin)
      if ratioContent > 0 :
        hRatio.SetBinError(ibin, ratioContent * dataError / dataContent) 
  
    
    # Code to draw error bars even if point outside range 
    # hRatio.Draw("PE same")
    oldSize = hRatio.GetMarkerSize()
    hRatio.SetMarkerSize(0)
    hRatio.DrawCopy("same e0")
    hRatio.SetMarkerSize(oldSize)
    hRatio.Draw("PE same") 
    hRatio.GetYaxis().SetTickSize(0)
  
    ytitle = 'Data / SM'
    customise_axes(h_mcErrRatio, xtitle, ytitle, 1.2)
    gPad.RedrawAxis() 
    
    # Insert arrows indicating data point is out of range of ratio panel
    l_arrows = {} 
    for mybin in range( hRatio.GetXaxis().GetNbins()+1 ):  
      Rdat = hRatio.GetBinContent(mybin)
      xval = hRatio.GetBinCenter(mybin)
      # print( 'Rdat: {0}, xval: {1}'.format(Rdat, xval) )
      if Rdat > 2:
        l_arrows[xval] = cut_arrow( xval, 1.7, xval, 1.9, 'up', 0.008, 6, kOrange+2 )
        l_arrows[xval][1].Draw()
    # customise_axes(hRatio, xtitle, ytitle, 1.2)
    # draw_data_vs_mc(h_dat, h_bkg, h_mcErr, xtitle, hXmin, hXmax) 
    #==========================================================
  
  #==========================================================
  # save everything
  ROOT.gStyle.SetLineScalePS(2) # to make all lines in the plot thinner, 3L analysis
  can.cd()
  can.SaveAs(save_name + '.pdf')
  #can.SaveAs(save_name + '.eps')
  #can.SaveAs(save_name + '.png')
  can.Close()

#____________________________________________________________________________
def draw_sig_scan(l_signals, d_hsigZ, cut_dir, xtitle, hXmin, hXmax):
  '''
  Draw significance scan 
  for signals in list l_signals
  using significance histograms d_hsigZ
  labelled by cut_dir, xtitle
  in range hXmin, hXmax
  '''
  print('Making significance scan plot in lower panel')
  #----------------------------------------------------
  # Draw significances
  d_samp = configure_samples(isData15_16, isData17, isData18, isData15_18, Signal)
  ytitle = 'Significance Z'
  for i, samp in enumerate(l_signals):
    hsigZ = d_hsigZ[samp]
    hsigZ.Draw('hist same')
    if i < 1:
      customise_axes(hsigZ, xtitle, ytitle, 1.2)
    l_color     = d_samp[samp]['l_color'] 
    format_hist(hsigZ, 3, l_color, 7, 0) # 3 = line thickhness, 7 = line style (long dashes), 3L analysis
    # format_hist(hsigZ, 2, l_color, 2, 0)  

  # Draw line for the ratio = 1
  l = draw_line(hXmin, 1.97, hXmax, 1.97, color=kAzure+1, style=7) 
  l.Draw()
  if 'upper' in cut_dir: # 3L analysis (upper and lower instad of left and right)
    myText(0.77, 0.83, 'Upper cut',  0.07, kBlack)
  if 'lower' in cut_dir:
    myText(0.77, 0.83, 'Lower cut', 0.07, kBlack)
  # if 'left' in cut_dir:
  #   myText(0.50, 0.83, 'Cut left, Z(s #geq 3, b #geq 1, #Delta b/b = 30%)',  0.07, kBlack)
  # if 'right' in cut_dir:
  #   myText(0.50, 0.83, 'Cut right, Z(s #geq 3, b #geq 1, #Delta b/b = 30%)', 0.07, kBlack)


#____________________________________________________________________________
def mk_sigZ_plot(h_intgl_sig, h_intgl_bkg, pc_syst, Nbins=100, xmin=0, xmax=100):
  '''
  Takes background & signal one-sided integral histograms
  and input percentage systematic
  Returns the signal significance Z histogram
  '''
  print('Making significance plot')
  h_pcsyst = TH1D('', "", Nbins, xmin, xmax)
  h_05syst = TH1D('', "", Nbins, xmin, xmax)
  h_20syst = TH1D('', "", Nbins, xmin, xmax)
  for my_bin in range( h_intgl_bkg.GetStack().Last().GetSize() ): 
    sExp     = h_intgl_sig.GetBinContent(my_bin)
    bExp     = h_intgl_bkg.GetStack().Last().GetBinContent(my_bin)  
    bin_low  = h_intgl_bkg.GetStack().Last().GetBinLowEdge( my_bin )
   
    # Case pathology 
    # Set significance is 0 if bExp or sExp is below 0
    if bExp <= 0 or sExp <= 0: 
    # if bExp < 1 or sExp < 3:
      RS_sigZ = 0
    else: 
      
      # Add statistical and systematic uncertainties in quadrature
      # BUnc   = sqrt ( abs( bExp + ( ( pc_syst / float(100) ) * bExp ) ** 2 ) )
      # RS_sigZ = RooStats.NumberCountingUtils.BinomialExpZ( sExp, bExp, BUnc/float(bExp) ) # Wrong? Inputs both stat and syst uncert to BinimialExpZ function, but stat uncert already taken care of
 
      # Only give (relative) systematic uncertainty as third argument to the Z_N function
      RS_sigZ = RooStats.NumberCountingUtils.BinomialExpZ( sExp, bExp, pc_syst / float(100)) 
      h_pcsyst.Fill(bin_low, RS_sigZ)
      # print('{0}, {1}, {2}, {3}, {4}, {5}'.format(my_bin, bin_low, bExp, sExp, my_sigZ, RS_sigZ) )
      # BUnc05 = sqrt ( abs( bExp + ( 0.05 * bExp ) ** 2 ) )
      # BUnc20 = sqrt ( abs( bExp + ( 0.20 * bExp ) ** 2 ) )
      # Calculate my significance
      # my_sigZ = sExp / float( BUnc )
   
  return h_pcsyst

#____________________________________________________________________________
def mk_mcErr(hStack, pc_sys, Nbins=100, xmin=0, xmax=100, variable_bin=False, hXarray=0):
  '''
  smear stacked MC histogram with 'Gaussian' sqrt(N) to emulate stats 
  also add a pc systematic
  '''
  if variable_bin:
    h_mcErr = TH1D('', "", Nbins, array('d', hXarray) )
  else:
    h_mcErr = TH1D('', "", Nbins, xmin, xmax)
  
  print( 'Making MC err' )
  for my_bin in range( hStack.GetStack().Last().GetSize() ):
    yval = hStack.GetStack().Last().GetBinContent(my_bin)
    
    if yval == 0:
      yval = 0.001
    # ============================================================ 
    # SERIOUS ISSUE: NEED TO INVESTIGATE!
    # why are there negative histogram values? something flawed going on
    # for now, take mean of adjacent bin y-values to disguise anomaly
    if yval < 0:
      yval = 0.001
      print( '\nSERIOUS WARNING: negative histogram value {0} in bin {1}'.format(yval, my_bin) )
      print('Please investigate. For now setting value to 0.001.') 
      # print( 'Please investigate. For now setting value to mean of neighbouring bins.\n' )
      # yMinus1 = hStack.GetStack().Last().GetBinContent(my_bin - 1)
      # yPlus1  = hStack.GetStack().Last().GetBinContent(my_bin + 1)
      # yval = (yPlus1 + yMinus1) / float(2)
    # ============================================================ 
  
    # Get statistical variance as sum of weights squared
    yval_GetErr   = hStack.GetStack().Last().GetBinError(my_bin)
    # Add stat and sys err in quadrature
    yval_err = sqrt( yval_GetErr ** 2 + ( 0.01 * pc_sys * yval ) ** 2 )
    
    # Fill histogram
    h_mcErr.SetBinContent( my_bin, yval )
    h_mcErr.SetBinError(   my_bin, yval_err ) 
  
  return h_mcErr
   
#_______________________________________________________
def tree_get_th1f(group, samps, files, var, cutsAfter='', Nbins=100, xmin=0, xmax=100, lumifb=35, variable_bin=False, hXarray=0, showOverflow=True): # 3L analysis
#def tree_get_th1f(f, hname, var, cutsAfter='', Nbins=100, xmin=0, xmax=100, lumifb=35, variable_bin=False, hXarray=0, showOverflow=True):

  '''
  from a TTree, project a leaf 'var' and return a TH1F
  '''

  # Initialise histogram
  #  --------------------
  if variable_bin:
    h_AfterCut   = TH1D(group + '_hist', "", Nbins, array('d', hXarray) ) # 3L analysis
    #h_AfterCut   = TH1D(hname + '_hist', "", Nbins, array('d', hXarray) )
    #hOneBin      = TH1D(hname + '_onebin', '', 2, 0, 2)
  else:
    h_AfterCut   = TH1D(group + '_hist', "", Nbins, xmin, xmax) # 3L analysis
    #h_AfterCut   = TH1D(hname + '_hist', "", Nbins, xmin, xmax) 
    #hOneBin      = TH1D(hname + '_onebin', '', 2, 0, 2)

  h_AfterCut.Sumw2() # Creates structure to store sum of squares of weights
  
  # Luminosity and weights
  # ---------------------- 
  lumi = lumifb*1000 # 3L analysis
  # lumi = "139000" # 3L analysis

  # if (RandomRunNumber < 320000): to be used with data15-16, lumi = 36.2 /fb
  # else if (RandomRunNumber > 320000 && RandomRunNumber > 348000): to be used with data17, lumi = 44.3 /fb
  # else if (RandomRunNumber > 348000): to be used with data18, lumi = 59.9 /fb
 
  # if isData15_16:
  #   lumi     = "36200"
  # elif isData17:
  #   lumi     = "44300"
  # elif isData18:
  #   lumi     = "59900"
  # elif isData15_18:
  #   lumi     = "139000"                                                        
  #else:
  #  lumi     = "RandomRunNumber < 320000 ? 36200 : ((RandomRunNumber > 320000 && RandomRunNumber < 348000) ? 44300 : 59900 )"
  
  if ('LFCMN1150' in group) or ('LFCMN1450' in group):
    weights = "genWeight * leptonWeight * bTagWeight * jvtWeight * ((DatasetNumber==364283 && RunNumber == 310000 && pileupWeight>2) ? 1 : pileupWeight)"  # FFWeight * triggerWeight*eventWeight
  else:
    weights = "genWeight * leptonWeight * eventWeight * bTagWeight * jvtWeight * ((DatasetNumber==364283 && RunNumber == 310000 && pileupWeight>2) ? 1 : pileupWeight)"  # FFWeight * triggerWeight

  if 'diboson3L' in group: # Histfitter NF for WZ
    if '0J' in sig_reg:
      weights = weights + ' * 1.0297' 
    elif 'nJ' in sig_reg:
      weights = weights + ' * 0.91145'

  # Weighted cut string   
  cut_after = '({0}) * {1} * ({2})'.format(cutsAfter, weights, lumi) 
  #print(cut_after) 

  # Initialise, fill and project TChain
  # -----------------------------------
  # Add TTrees from different group samples (periods) to a TChain, 
  # and then fill histogram using TTree::Project() on the TChain

  # 3L analysis
  chain = TChain( group ) 
  if ntuple_version == 'v2.3c':
    if group.startswith('MGPy8EG'): # signal  #LOOSE_NOMINAL
      chain.AddFile( files[0] + '/{}_NoSys'.format(samps[0]) )
    elif 'data' in group: # data 
      chain.AddFile( files[0] + '/{}'.format(samps[0]) )
    else: # bkg    #LOOSE_NOMINAL
      chain.AddFile( files[0] + '/{}_NoSys'.format(samps[0]) )
      if len(files) > 1:
        for i in range(1, len(files)):   #LOOSE_NOMINAL
          chain.AddFile( files[i] + '/{}_NoSys'.format(samps[i]) )

  """
  if 'data' not in hname:
    chain = TChain( hname + '_NoSys' )
  elif 'data' in hname:
    chain = TChain( hname )

  if isData15_16 or isData15_18:
    chain.Add( f[0] )
  if isData17 or isData15_18:
      chain.Add( f[1] )
  if isData18 or isData15_18:
    chain.Add( f[2] )
  """ 

  # 3L analysis
  if 'data' not in group:
    chain.Project( group + '_hist', var, cut_after )
  elif 'data' in group:
    chain.Project( group + '_hist', var, cutsAfter )

  """
  if 'data' not in hname:
    chain.Project( hname + '_hist', var, cut_after )
    #chain.Project( hname + '_onebin', 'lepSignal[0]', cut_after )
  elif 'data' in hname:
    chain.Project( hname + '_hist', var, cutsAfter )
    #chain.Project( hname + '_onebin', 'lepSignal[0]', cutsAfter )
  """
  
  # Perform integrals
  # -----------------
  # Find total yield, one-sided lower and upper cumulative histos
  nYieldErr = ROOT.Double(0)  
  nYield    = h_AfterCut.IntegralAndError(0, Nbins+1, nYieldErr)

  h_intgl_lower = TH1D(group + '_intgl_lower', "", Nbins, xmin, xmax) # 3L analysis
  h_intgl_upper = TH1D(group + '_intgl_upper', "", Nbins, xmin, xmax) # 3L analysis
  # h_intgl_lower = TH1D(hname + '_intgl_lower', "", Nbins, xmin, xmax)
  # h_intgl_upper = TH1D(hname + '_intgl_upper', "", Nbins, xmin, xmax)

  # h_intgl_overflow = TH1D(hname + '_intgl_overflow', "", Nbins, xmin, xmax)
  # h_intgl_overflow.Sumw2()
  
  for my_bin in range( h_AfterCut.GetXaxis().GetNbins() + 1 ):
    
    # Get lower edge of bin
    bin_low = h_AfterCut.GetXaxis().GetBinLowEdge( my_bin )
    
    # Set the negatively weighted values to 0.
    bin_val = h_AfterCut.GetBinContent( my_bin )
    if bin_val < 0:
      print( 'WARNING: Bin {0} of sample {1} has negative entry, setting central value to 0.'.format(my_bin, group) ) # 3L analysis
      # print( 'WARNING: Bin {0} of sample {1} has negative entry, setting central value to 0.'.format(my_bin, hname) )
      h_AfterCut.SetBinContent(my_bin, 0.)
    
    # Do one-sided integral either side of bin<
    intgl_lower = h_AfterCut.Integral( 0, my_bin ) 
    intgl_upper = h_AfterCut.Integral( my_bin, Nbins+1 ) 
    
    h_intgl_lower.Fill( bin_low, intgl_lower )
    h_intgl_upper.Fill( bin_low, intgl_upper )

  h_intgl_overflow = h_AfterCut.Clone("h_intgl_overflow")
  content_binN = h_intgl_overflow.GetBinContent(Nbins)
  content_overflowBin = h_intgl_overflow.GetBinContent(Nbins+1)
  h_intgl_overflow.SetBinContent(Nbins, content_binN + content_overflowBin)
  h_intgl_overflow.SetBinContent(Nbins+1, 0.)
  nYieldErr_overflow = ROOT.Double(0)
  nYield_overflow = h_intgl_overflow.IntegralAndError(0, Nbins+1, nYieldErr_overflow)

  print( 'Sample {0} has integral {1:.3f} +/- {2:.3f} \n'.format( group, nYield, nYieldErr ) ) # 3L analysis
  # print( 'Sample {0} has integral {1:.3f} +/- {2:.3f} \n'.format( hname, nYield, nYieldErr ) )
  # print( 'Sample {0} has integral {1:.3f} +/- {2:.3f} (overflow histo integral {1:.3f} +/- {2:.3f})'.format( hname, nYield, nYieldErr, nYield_overflow, nYieldErr_overflow ) )
  # =========================================================

  if showOverflow:
    h_AfterCut = h_intgl_overflow
  
  return [h_AfterCut, nYield, h_intgl_lower, h_intgl_upper, nYieldErr, h_intgl_overflow, nYield_overflow]

#____________________________________________________________________________
def format_hist(hist, l_width=2, l_color=kBlue+2, l_style=1, f_color=0, f_style=1001, l_alpha=1.0):
  
  # Lines
  hist.SetLineColorAlpha(l_color, l_alpha)
  hist.SetLineStyle(l_style)
  hist.SetLineWidth(l_width)
  hist.SetLineColor(l_color) # In order to get contours for the individual bkg hists, 3L analysis

  # Fills
  hist.SetFillColor(f_color)
  hist.SetFillStyle(f_style)

  # Markers
  hist.SetMarkerColor(l_color)
  hist.SetMarkerSize(1.1)
  hist.SetMarkerStyle(20)

#____________________________________________________________________________
def customise_gPad(top=0.03, bot=0.15, left=0.17, right=0.08):
  gPad.Update()
  gStyle.SetTitleFontSize(0.0)
  
  # gPad margins
  gPad.SetTopMargin(top)
  gPad.SetBottomMargin(bot)
  gPad.SetLeftMargin(left)
  gPad.SetRightMargin(right)
  
  gStyle.SetOptStat(0) # Hide usual stats box 
  
  gPad.Update()
  
#____________________________________________________________________________
def customise_axes(hist, xtitle, ytitle, scaleFactor=1.1, IsLogY=False, enlargeYaxis=False):
  # Set a universal text size
  # text_size = 0.055
  text_size = 45
  TGaxis.SetMaxDigits(4) 
  ##################################
  # X axis
  xax = hist.GetXaxis()
  
  # Precision 3 Helvetica (specify label size in pixels)
  xax.SetLabelFont(43)
  xax.SetTitleFont(43)
  # xax.SetTitleFont(13) # times
  
  xax.SetTitle(xtitle)
  xax.SetTitleSize(text_size)
  # Top panel
  # if xtitle == '':
  if 'Events' in ytitle:
  # if False:
    xax.SetLabelSize(0)
    xax.SetLabelOffset(0.02)
    xax.SetTitleOffset(2.0)
    xax.SetTickSize(0.04)  
  # Bottom panel
  else:
    xax.SetLabelSize(text_size - 7)
    xax.SetLabelOffset(0.03)
    xax.SetTitleOffset(3.5)
    xax.SetTickSize(0.08)
 
  # xax.SetRangeUser(0,2000) 
  # xax.SetNdivisions(-505) 
  gPad.SetTickx() 
  
  ##################################
  # Y axis
  yax = hist.GetYaxis()
  # Precision 3 Helvetica (specify label size in pixels)
  yax.SetLabelFont(43)
  yax.SetTitleFont(43)
 
  yax.SetTitle(ytitle)
  yax.SetTitleSize(text_size)
  yax.SetTitleOffset(1.8)    
  
  yax.SetLabelOffset(0.015)
  yax.SetLabelSize(text_size - 7)
 
  ymax = hist.GetMaximum()
  ymin = hist.GetMinimum()
  
  # Top events panel
  # if xtitle == '':
  if 'Events' in ytitle:
    yax.SetNdivisions(505) 
    if IsLogY:
      if enlargeYaxis:
        ymax = 2 * 10 ** 10
        ymin = 0.02
      else:
        # ymax = 3 * 10 ** 4
        # ymin = 0.5
        ymax = 3 * 10 ** 3
        ymin = 0.005
      hist.SetMaximum(ymax)
      hist.SetMinimum(ymin)
    else:
      hist.SetMaximum(ymax*scaleFactor)
      # hist.SetMaximum(100)
      # hist.SetMaximum(30)
      # hist.SetMaximum(60)
      hist.SetMinimum(0.0)
  # Bottom data/pred panel 
  elif 'Significance' in ytitle:
    hist.SetMinimum(-0.5) # Sigaxis
    hist.SetMaximum(5.0)  # Sigaxis
    yax.SetNdivisions(205)
  elif 'Data' in ytitle:
    hist.SetMinimum(0.0)
    hist.SetMaximum(2.0) 
    yax.SetNdivisions(205)
   
  gPad.SetTicky()
  gPad.Update()

#____________________________________________________________________________
def myText(x, y, text, tsize=0.05, color=kBlack, angle=0) :
  
  l = TLatex()
  l.SetTextSize(tsize)
  l.SetNDC()
  l.SetTextColor(color)
  l.SetTextAngle(angle)
  l.DrawLatex(x,y,'#bf{' + text + '}')
  l.SetTextFont(4)

#____________________________________________________________________________
def cut_arrow(x1, y1, x2, y2, direction='lower', ar_size=1.0, ar_width=10, color=kGray+3, style=1) :
  
  l = TLine(x1, y1, x2, y2)
  if direction == 'lower':
    ar = TArrow(x1-0.02, y2, x1+ar_width, y2, ar_size, '|>')
  if direction == 'upper':
    ar = TArrow(x1-ar_width+0.02, y2, x1, y2, ar_size, '<|')
  if direction == 'up':
    ar = TArrow(x1, y1, x1, y2, ar_size, '|>')
  if direction == 'left': #'down'
    ar = TArrow(x1, y1, x1, y2, ar_size, '<|')
  l.SetLineWidth(4)
  l.SetLineStyle(style)
  l.SetLineColor(color) 
  ar.SetLineWidth(4)
  ar.SetLineStyle(style)
  ar.SetLineColor(color) 
  ar.SetFillColor(color)  
  return [l, ar]

#____________________________________________________________________________
def mk_leg(xmin, ymin, xmax, ymax, sig_reg, dict_samp, d_samp, nTotBkg, d_hists, d_yield, d_yieldErr, yields_file, sampSet_type='bkg', txt_size=0.05) : # 3L analysis 
#def mk_leg(xmin, ymin, xmax, ymax, sig_reg, l_samp, d_samp, nTotBkg, d_hists, d_yield, d_yieldErr, sampSet_type='bkg', txt_size=0.05) :
  '''
  @l_samp : Constructs legend based on list of samples 
  @nTotBkg : Total background events
  @d_hists : The dictionary of histograms 
  @d_samp : May from samples to legend text
  @d_yields : The dictionary of yields 
  @d_yieldErr : Dictionary of errors on the yields
  @sampSet_type : The type of samples in the set of samples in the list 
  '''  

  # ---------------------------------------------------- 
  # Legend: construct and format
  # ---------------------------------------------------- 
  leg = TLegend(xmin,ymin,xmax,ymax)
  leg.SetBorderSize(0)
  leg.SetTextSize(txt_size)
  leg.SetNColumns(1)

  # Legend markers 
  d_legMk = {
    'bkg'  : 'f',
    'sig'  : 'l',
    'data' : 'ep'
  }

  # Need to reverse background order so legend is filled as histogram is stacked
  #if sampSet_type == 'bkg':
  #l_samp = [x for x in reversed(l_samp)]

  # Legends written in the same order as the histograms (i.e. ordered after yields)             
  for samp in reversed(sorted(dict_samp.keys(), key=lambda s:(d_hists[s][1]))): # 3L analysis
  #for samp in l_samp: 
    #print( 'Processing {0}'.format(samp) )
    # obtain sample attributes 
    hist        = d_hists[samp][0]
    sample_type = d_samp[samp]['type']
    leg_entry   = d_samp[samp]['leg']
    legMk       = d_legMk[sample_type]
   
    #print('samp: {0}, type: {1}, legMk: {2}'.format(samp, sample_type, legMk) ) 

    # Calculate the % of each background component and put in legend
    pc_yield   = 0
    if sample_type == 'bkg':
      pc_yield = 100 * ( d_yield[samp] / float(nTotBkg) )
      leg_txt = '{0} ({1:.1f}, {2:.1f}%)'.format( leg_entry, d_yield[samp], pc_yield ) 
      #leg_txt = '{0} ({1:.1f}%)'.format( leg_entry, pc_yield )
    if sample_type == 'sig':
      leg_txt = '{0} ({1:.1f})'.format(leg_entry, d_yield[samp])
    if sample_type == 'data':
      leg_txt = '{0} ({1:.0f})'.format(leg_entry, d_yield['data'])
      # leg_txt = '{0} ({1:.0f} Events)'.format(leg_entry, d_yield['data'])  

    leg.AddEntry(hist, leg_txt, legMk)
    print('{0}, {1}, {2:.1f} +/- {3:.1f}'.format(sig_reg, samp, d_yield[samp], d_yieldErr[samp]) )
    #print('{0}, {1}, {2:.3f}, {3:.3f}%'.format(sig_reg, samp, d_yield[samp], pc_yield) )

    # Save yield to file
    if allNCuts:
      yields_file.write(samp+'\t'+str(d_yield[samp])+'\t'+str(d_yieldErr[samp])+'\n') 

  return leg

#____________________________________________________________________________
def draw_line(xmin, ymin, xmax, ymax, color=kGray+1, style=2) :
  
  # Draw line of kinematically forbidden region
  line = TLine(xmin , ymin , xmax, ymax)
  line.SetLineWidth(2)
  line.SetLineStyle(style)
  line.SetLineColor(color) # 12 = gray
  return line

#_________________________________________________________________________
def mkdir(dirPath):
  '''
  make directory for given input path
  '''
  try:
    os.makedirs(os.path.expandvars(dirPath))
    print('Successfully made new directory ' + dirPath)
  except OSError:
    pass


#====================
# if name = main
#=================== 
if __name__ == "__main__":
  #main(sys.argv)
  main()        
