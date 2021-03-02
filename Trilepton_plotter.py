import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


read_DF = pd.read_hdf("Trilepton_ML.h5", key="DF_flat3")
#print(read_DF.info())


### Functions for plotting various variables in the dataframe: ###
def plotter(var):
    """Plots chosen variable property for all the leptons in one plot."""
    try:
        df = read_DF[["lep1_"+var, "lep2_"+var, "lep3_"+var, "lep4_"+var]]

        df.plot.hist(stacked=False, histtype="step", bins=100, log=True)
        plt.title("Plot of "+var+" for each lepton")
        plt.show()
    except:
        print("Variable is not defined in dataframe! Choose another:")
        print(read_DF.columns)


def sub_plotter(var):
    """Plots a set of chosen variables which are connected to two leptons at a time, where each subplot shows the connection of the i'th lepton to the other leptons."""
    fig, axes = plt.subplots(4, 1, figsize=(12,8), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flatten()

    if var == "mll":
        fig.suptitle("Invariant mass of lepton -i- and -j- in mll_ij", fontsize=16)
    elif var == "dphi":
        fig.suptitle("Azimuthal angle between lepton -i- and -j- in dPhi_ij", fontsize=16)
    elif var == "dR":
        fig.suptitle("Angular distance between lepton -i- and -j- in dR_ij", fontsize=16)
    else:
        print("Choose a variable in the dataframe:")
        print(read_DF.columns)
        sys.exit()

    df_1 = read_DF[[var+"_12", var+"_13", var+"_14"]]
    df_2 = read_DF[[var+"_12", var+"_23", var+"_24"]]
    df_3 = read_DF[[var+"_13", var+"_23", var+"_34"]]
    df_4 = read_DF[[var+"_14", var+"_24", var+"_34"]]

    df_1.plot.hist(ax=ax0, stacked=False, histtype="step", bins=100, log=True)
    df_2.plot.hist(ax=ax1, stacked=False, histtype="step", bins=100, log=True)
    df_3.plot.hist(ax=ax2, stacked=False, histtype="step", bins=100, log=True)
    df_4.plot.hist(ax=ax3, stacked=False, histtype="step", bins=100, log=True)
    
    plt.show()


""" 
Available variables to plot:
    -plotter (10): pt, phi, eta, theta, px, py, pz, E, vtx, pid
    -sub_plotter (3): mll, dphi, dR
"""

plotter("pt")
#sub_plotter("mll")
#sub_plotter("dphi")
#sub_plotter("dR")
