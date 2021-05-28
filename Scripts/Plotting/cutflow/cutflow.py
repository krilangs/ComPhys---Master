#!/usr/bin/env python
'''
plot.py is the main script to do the plotting
This reads the ntuples produced by SusySkimHiggsino
Makes plots of data vs MC in various variables
Configure various aspects in
  - cuts.py
  - samples.py
  - variables.py
One specifies the samples to be plotted at the top of calc_selections() function
'''

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # So Root ignores command line inputs so we can use argparse
ROOT.gROOT.SetBatch(1) # No graphics
from ROOT import *

import os, sys, time, argparse

from math import sqrt
from random import gauss
from array import array

from samples_cutflow import *
from cuts_cutflow import *
from variables import *

import cut_labels

# Path to directory of full ntuples  
TOPPATH = '/eos/atlas/atlascerngroupdisk/phys-susy/EWK3L_ANA-SUSY-2018-06/ntuples/offshell_v2.3c/allSys_fullJES' # Reco samples

lumi_15_18 = 139
annotate_text = '2015-18 data vs. mc16a+d+e'  
text_size = 0.045 # Percentage
savedir_cutflow = './output/'  



#____________________________________________________________________________
def main():
  
  # Keep track of time spent
  t0 = time.time()

  # Initialize global variables
  global sig_reg, savedir, lumi 

  # Default values
  var     = 'met_Et'
  sig_reg = 'SRhigh_0Jb'
  savedir = savedir_cutflow
  lumi    = lumi_15_18
  unblind = False
  showOverflow = True
  
  # Check if user has inputted variables or not
  parser = argparse.ArgumentParser(description='Analyse background/signal TTrees and make plots.')
  parser.add_argument('-v', '--variable',  type=str, nargs='?', help='String name for the variable to make N-1 in. Either as appearing in TTree, or, if added, with additional plot information', default=var) # Option for the var "name" to include plotting information has been added for 3L analysis
  parser.add_argument('-s', '--sigReg',    type=str, nargs='?', help='String name of selection (signal/control) region to perform N-1 selection.', default=sig_reg)
  parser.add_argument('-l', '--lumi',      type=str, nargs='?', help='Float of integrated luminosity to normalise MC to.', default=lumi)
  parser.add_argument('-u', '--unblind',   type=str, nargs='?', help='Should the SRs be unblinded?')
  parser.add_argument('-o', '--notShowOverflow',  action='store_true', help='Do not include overflow in bin N.')
  args = parser.parse_args()
  if args.variable:
    var      = args.variable
  if args.sigReg:
    sig_reg = args.sigReg
  if args.lumi:
    lumi = args.lumi
  if args.unblind == 'True':
    unblind = True
  if args.notShowOverflow:
    showOverflow = False

  # Print summary to screen
  print( '\n=========================================' )
  print( 'Samples:    {0}'.format(TOPPATH) )
  print( 'Region:     {0}'.format(sig_reg) )
  print( 'Luminosity: {0}'.format(lumi) )
  print( 'Unblind:    {0}'.format(unblind) )
  print( '=========================================\n' )
  
  # Make (relative) save directory if needed 
  mkdir(savedir)

  # List samples to analyse 
  calc_selections(var, lumi, sig_reg, unblind, showOverflow)
  
  # Keep track of time spent
  tfinish = time.time()
  telapse = tfinish - t0
  m, s = divmod(telapse, 60)  
  print('{} min {} s'.format(int(m), int(round(s))))



#____________________________________________________________________________
def calc_selections(var, lumi, sig_reg, unblind=False, showOverflow=True):
  '''
  Extract trees given a relevant variable
  '''
  #==========================================================
  # Prepare information and objects for analysis and plots
  #==========================================================

  d_samp_bkg = { 
    # 'other'     : ['diboson2L','singletop'],
    # 'fakes'     : ['fakes'],
    # 'ttbar'     : ['ttbar'],
    # 'ttbar+X'   : ['topOther','higgs'],
    # 'triboson'  : ['triboson'],
    # 'diboson4L' : ['diboson4L'],
    'diboson3L' : ['diboson3L']
   }

  d_samp_signal = { 
    
    # 3L3
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_10_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_10_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_20_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_20_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_30_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_30_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_40_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_40_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_60_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_60_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_80_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_80_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_90_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_90_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_100_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_100_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_20_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_20_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_30_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_30_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_40_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_40_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_50_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_50_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_70_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_70_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_85_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_85_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_95_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_95_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_100_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_100_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_110_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_110_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_115_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_115_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_35_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_35_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_45_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_45_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_55_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_55_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_65_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_65_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_85_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_85_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_130_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_130_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_50_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_50_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_60_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_60_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_70_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_70_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_140_80_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_140_80_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_100_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_100_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_110_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_110_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_120_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_120_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_130_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_130_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_140_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_140_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_105_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_105_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_108_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_108_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3_NoSys'],
      'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_170_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_170_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3_NoSys'],
      'MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_155_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_155_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_158_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_158_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_160_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_160_3L_3L3_NoSys'],
      'MGPy8EG_A14N23LO_C1N2_WZ_250_170_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_170_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_190_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_190_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_200_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_200_3L_3L3_NoSys'],
      'MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_220_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_220_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_230_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_230_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_240_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_240_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_208_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_220_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
      'MGPy8EG_A14N23LO_C1N2_WZ_300_240_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_260_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_280_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_300_290_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_350_270_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_350_270_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_350_290_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_350_290_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_350_310_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_350_310_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_350_330_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_350_330_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_350_340_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_350_340_3L_3L3_NoSys'],

    # 'MGPy8EG_A14N23LO_C1N2_WZ_400_320_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_400_320_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_400_340_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_400_340_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_400_360_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_400_360_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_400_380_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_400_380_3L_3L3_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_400_390_3L_3L3_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_400_390_3L_3L3_NoSys'],

    # # 3L2MET75
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_95_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_95_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_97_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_97_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_105_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_105_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_110_107_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_110_107_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_120_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_120_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_125_122_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_125_122_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_145_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_145_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_147_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_147_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_175_170_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_175_170_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_175_172_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_175_172_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_195_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_195_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_197_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_197_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_90_85_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_90_85_3L2MET75_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_90_87_3L2MET75_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_90_87_3L2MET75_NoSys'],

    # # 2L7
    # 'MGPy8EG_A14N23LO_C1N2_WZ_100_0_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_100_0_3L_2L7_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_1_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_1_3L_2L7_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_150_50_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_150_50_3L_2L7_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_100_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_100_3L_2L7_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_200_50_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_200_50_3L_2L7_NoSys'],
    # 'MGPy8EG_A14N23LO_C1N2_WZ_250_150_3L_2L7_NoSys' : ['MGPy8EG_A14N23LO_C1N2_WZ_250_150_3L_2L7_NoSys'],
  }

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

  # Obtain nested list of cuts (strings) for cutflow
  l_all_cuts, l_nested_cuts = configure_cuts(sig_reg)
  # Add no cuts scenario manually
  #l_nested_cuts.insert(0, "1") 

  # Get dictionary defining sample properties
  d_samp = configure_samples()  
  
  # Initialise objects to fill in loop 
  d_files = {}
  
  #==========================================================
  # Loop through samples, fill histograms
  #==========================================================

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
      full_path = TOPPATH + '/' + path

      # Store full paths in dictonary 
      d_files[group] += [full_path]

    # Obtain histogram from file and store to dictionary entry
    tree_get_th1f_cutflow( group, samples, d_files[group], var_plot, l_all_cuts, l_nested_cuts, hNbins, hXmin, hXmax, lumi, variable_bin, hXarray, showOverflow )
    

#_______________________________________________________
def tree_get_th1f_cutflow(group, samps, files, var, l_all_cuts, l_nested_cuts, Nbins=100, xmin=0, xmax=100, lumifb=35, variable_bin=False, hXarray=0, showOverflow=True): # 3L analysis

  '''
  from a TTree, project a leaf 'var' and return a TH1F
  '''
  
  # Luminosity, xsec, BF, filtEff and weights
  # ---------------------- 
  lumi = lumifb*1000 # Normalise yields to 1 pb-1 (since cross sections usually are given in pb)
  BF = 0.0327 # Not signal only, right? ->  3*BF(W->lv)*3*BF(Z->ll) 

  # For signal, read file to get xsec and filter efficiency
  if group.startswith('MGPy8EG'):
    mC1 = group.split('_')[4]
    mN1 = group.split('_')[5]

    offshell = True
    if (group.find('3L3')>-1):
      grid = '3L3'
      xsec_feff_file = open('./xsec_feff_files/3L3_xsec_feff.txt', 'r') # Cross sections (NB! unit nb) and filter efficiencies extracted from AMI
    elif (group.find('3L2MET75')>-1):
      grid = '3L2MET75'
      xsec_feff_file = open('./xsec_feff_files/3L2MET75_xsec_feff_PMG.txt', 'r') # Cross sections (NB! unit pb) and filter efficiencies extracted from PMG
    elif (group.find('2L7')>-1):
      grid = '2L7'
      offshell = False
      xsec_feff_file = open('./xsec_feff_files/2L7_xsec_feff_PMG.txt', 'r') # Cross sections (NB! unit pb) and filter efficiencies extracted from PMG
    
    lines = xsec_feff_file.readlines()
    xsec_feff_file.close()

    for line in lines:
      line = line.strip().split('\t')

      if offshell:

        if grid  == '3L3':
          # Minus
          if (line[0] == 'C1mN2' and mC1 == line[2].split('p')[0] and mN1 == line[3].split('p')[0]):
            filtEff_m = float(line[5])
            xsec_m = float(line[4]) * 1000. # NB! Contains BF. Original unit nb, converted to pb 
            xsec_m_woBF = xsec_m / BF # Without BF
          # Plus
          elif (line[0] == 'C1pN2' and mC1 == line[2].split('p')[0] and mN1 == line[3].split('p')[0]):
            filtEff_p = float(line[5])
            xsec_p = float(line[4]) * 1000. # NB! Contains BF. Original unit nb, converted to pb 
            xsec_p_woBF = xsec_p / BF # Without BF
        
        elif grid == '3L2MET75':
          # Minus
          if (line[0] == 'C1mN2' and mC1 == line[2] and mN1 == line[3]):
            filtEff_m = float(line[5])
            xsec_m = float(line[4]) # NB! Contains BF. Unit pb 
            xsec_m_woBF = xsec_m / BF # Without BF
          # Plus
          elif (line[0] == 'C1pN2' and mC1 == line[2] and mN1 == line[3]):
            filtEff_p = float(line[5])
            xsec_p = float(line[4]) # NB! Contains BF. Unit pb 
            xsec_p_woBF = xsec_p / BF # Without BF
        
      else: # onshell
    
        if (mC1 == line[2].split('p')[0] and mN1 == line[3].split('p')[0]):
          filtEff = float(line[5])
          xsec = float(line[4]) # NB! Contains BF. Unit pb
          xsec = xsec / BF # Without BF 

    # Combine plus and minus contributions in the correct way for offshell 
    if offshell:
      xsec = xsec_p_woBF + xsec_m_woBF 
      filtEff = ( filtEff_p * xsec_p_woBF + filtEff_m * xsec_m_woBF ) / xsec 

  #else: # background, what to do with xsec and filtEff
  #  xsec = 
  #  filtEff = 

  # All weights
  weights = "genWeight * leptonWeight * eventWeight * bTagWeight * jvtWeight * ((DatasetNumber==364283 && RunNumber == 310000 && pileupWeight>2) ? 1 : pileupWeight) * FFWeight * triggerWeight"
  if 'diboson3L' in group: # Histfitter NF for WZ
    if '0J' in sig_reg:
      weights = weights + ' * 1.0297' 
    elif 'nJ' in sig_reg:
      weights = weights + ' * 0.91145'

  # Fill TChain with samples
  # ----------------------
  chain = TChain( group ) 
  if group.startswith('MGPy8EG'):
    chain.AddFile( files[0] + '/{}'.format(samps[0]) )
  else: 
    chain.AddFile( files[0] + '/{}_NoSys'.format(samps[0]) )
    if len(files) > 1:
      for i in range(1, len(files)):
        chain.AddFile( files[i] + '/{}_NoSys'.format(samps[i]) )

  # Calculate and save yields cutflow style
  # ----------------------
  ########################################################################################
  # Prepare file for saving cutflow
  save_file_path = savedir + sig_reg + '_' + group + '_cutflow.txt'
  if os.path.exists(save_file_path):
    os.remove(save_file_path)
  save_file = open(save_file_path, 'a')

  ### First make cutflow for lumi, xsec, BF and filter efficiency, with no cuts (or weights!) added
  if group.startswith('MGPy8EG'): # NB!!! Remove when you know which XS and FE to use for bkgs

    #noCuts_list_strings = ['lumi x xsec', 'lumi x xsec x BF', 'lumi x xsec x BF x filtEff'] # For writing to file
    noCuts_list_strings = [ # For writing to file
      'Initial number of events $(\mathcal{L} '+r'\times \sigma)$', 
      'Initial number of events $(\mathcal{L} '+r'\times \sigma '+r'\times \mathcal{B})$',
      'Generator filters $(\mathcal{L} '+r'\times \sigma '+r'\times \mathcal{B} '+r'\times \epsilon)$',
    ] 
    noCuts_list = [lumi * xsec, lumi * xsec * BF, lumi * xsec * BF * filtEff]

    for j,elements in enumerate(noCuts_list):
      
      # Write yield to file
      save_file.write( '{0} & {1:.2f} \\\ \n'.format(noCuts_list_strings[j], noCuts_list[j]) )

  ### Then continue cutflow by adding cuts one by one
  for k,partial_cuts in enumerate(l_nested_cuts):

    # Initialise histogram
    if variable_bin:
      h_AfterCut   = TH1D(group + '_hist_' + str(k), "", Nbins, array('d', hXarray) ) 
    else:
      h_AfterCut   = TH1D(group + '_hist_' + str(k), "", Nbins, xmin, xmax) 
    h_AfterCut.Sumw2() # Creates structure to store sum of squares of weights

    # Weighted cut string
    cut_after = '({0}) * {1} * ({2})'.format(partial_cuts, weights, lumi) # NB! genWeight contains xsec, BF and filtEff
  
    # Fill histogram by projecting TChain
    if 'data' not in group:
      chain.Project( group + '_hist_' + str(k), var, cut_after )
    elif 'data' in group:
      chain.Project( group + '_hist_' + str(k), var, partial_cuts ) # No lumi or weights for data, only cuts
  
    # Perform integral to get total yield
    nYieldErr = ROOT.Double(0)  
    nYield    = h_AfterCut.IntegralAndError(0, Nbins+1, nYieldErr)

    # Write yield to tile
    #save_file.write('{0} \t {1:.2f} +/- {2:.2f} \n'.format( l_all_cuts[k], nYield, nYieldErr ) )
    #save_file.write('{0} \t {1:.2f} +/- {2:.2f} \n'.format( cut_labels.labels[l_all_cuts[k]], nYield, nYieldErr ) )
    #save_file.write('{0} & {1:.2f} $\pm$ {2:.2f} \\\ \n'.format( cut_labels.labels[l_all_cuts[k]], nYield, nYieldErr ) )
    save_file.write('{0} & {1:.2f} \\\ \n'.format( cut_labels.labels[l_all_cuts[k]], nYield, nYieldErr ) )
  ########################################################################################



#_________________________________________________________________________
def mkdir(dirPath):
  '''
  make directory for given input path
  '''
  try:
    os.makedirs(os.path.expandvars(dirPath))
    print 'Successfully made new directory ' + dirPath
  except OSError:
    pass



#_________________________________________________________________________
if __name__ == "__main__":
  #main(sys.argv)
  main()        
