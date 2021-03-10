import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

save_Folder = "/scratch2/Master_krilangs/Trilepton_Plots/Feature_plots/"

read_DF = pd.read_hdf("Trilepton_ML.h5", key="DF_flat3")
#print(read_DF.info())


### Functions for plotting various variables in the dataframe: ###
def plotter(var, Nbins=80, save=False):
    """Plots histogram of chosen variable property for all the leptons in one plot. If a variable is not in the dataframe, an error is raised and the variables available in the dataframe are printed."""
    try:
        df = read_DF[["lep1_"+var, "lep2_"+var, "lep3_"+var, "lep4_"+var]]

        if var == "phi":
            fig, axes = plt.subplots(4, 1, figsize=(10,6), constrained_layout=True)
            ax0, ax1, ax2, ax3 = axes.flatten()

            fig.suptitle("Histogram of "+var+" for each lepton", fontsize=16)
            df.lep1_phi.plot.hist(ax=ax0, histtype="step", bins=Nbins, log=True, label="lep1_phi", color="blue")
            df.lep2_phi.plot.hist(ax=ax1, histtype="step", bins=Nbins, log=True, label="lep2_phi", color="orange")
            df.lep3_phi.plot.hist(ax=ax2, histtype="step", bins=Nbins, log=True, label="lep3_phi", color="green")
            df.lep4_phi.plot.hist(ax=ax3, histtype="step", bins=Nbins, log=True, label="lep4_phi", color="red")
            
            ax0.legend(bbox_to_anchor=(1,0.5), loc="center left")
            ax1.legend(bbox_to_anchor=(1,0.5), loc="center left")
            ax2.legend(bbox_to_anchor=(1,0.5), loc="center left")
            ax3.legend(bbox_to_anchor=(1,0.5), loc="center left")
            
            if save:
                print("Save figure")
                fig.savefig(save_Folder+var+".png")

        else:
            df.plot.hist(stacked=False, histtype="step", bins=Nbins, log=True)
            plt.title("Histogram of "+var+" for each lepton")
            plt.tight_layout()
            if save:
                print("Save figure")
                fig.savefig(save_Folder+var+".png")

    except:
        print("If variable is not defined in dataframe, choose another:")
        print(read_DF.columns)
        raise

#-----
def sub_plotter(var, Nbins=80, save=False):
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

    df_1.plot.hist(ax=ax0, stacked=False, histtype="step", bins=Nbins, log=True)
    df_2.plot.hist(ax=ax1, stacked=False, histtype="step", bins=Nbins, log=True)
    df_3.plot.hist(ax=ax2, stacked=False, histtype="step", bins=Nbins, log=True)
    df_4.plot.hist(ax=ax3, stacked=False, histtype="step", bins=Nbins, log=True)

    if save:
        print("Save figure")
        fig.savefig(save_Folder+var+".png")


#-----
def plot_all(Nbins=80, save=False):
    """Plots all the variables, except vtx and pid since they are not as interesting."""
    singles = ["pt", "phi", "eta", "theta", "px", "py", "pz", "E"]
    multiples = ["mll", "dphi", "dR"]

    for i in range(len(singles)):
        plotter(singles[i], Nbins, save)

    for j in range(len(multiples)):
        sub_plotter(multiples[j], Nbins, save)

    
""" 
Available variables to plot:
    -plotter (10): pt, phi, eta, theta, px, py, pz, E, vtx, pid
    -sub_plotter (3): mll, dphi, dR
Both functions takes the following inputs as arguments:
    -var = str, The variable name.
    -Nbins = int, The number of bins.
    -save = bol, True; to save the plots to the folder at the start of the script.
"""

#plotter("pt", 80, False)
#plotter("phi", 80,  False)
#sub_plotter("mll", 80, False)
plot_all(80, False)

plt.show()
