import sys
import h5py
import uproot
import awkward
import itertools
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from math import sqrt, cos
import matplotlib.pyplot as plt
import uproot_methods.classes.TLorentzVector


perm = list(itertools.permutations([1, 2, 3]))
nupid = [-12, 12, -14, 14, -16, 16]

def get_invmass(value):
    #print("Type of value is ",type(value))
    try:
        return(value.mass)
    except:
        #print("Could not return mass for ", value)
        return np.nan #-1
#-----

def get_deltaPhi(x,y):
    #print("Type of value is ",type(x))
    try:
        return(x.delta_phi(y))
    except:
        print("Could not return deltaPhi for ", (x,y))
        return -1
#-----


def get_deltaR(x,y):
    #print(" of value is ",type(value))
    try:
        return(x.delta_r(y))
    except:
        print("Could not return deltaR for ", (x,y))
        return -1
#-----


def classify_event(l1,l2,l3,l4,p1,p2,p3,p4):
    if p1 in nupid: tuplex = l2,l3,l4
    elif p2 in nupid: tuplex = l1,l3,l4
    elif p3 in nupid: tuplex = l1,l2,l4
    elif p4 in nupid: tuplex = l1,l2,l3
    if tuplex in perm:
        return perm[perm.index(tuplex)]
    else:
        print("Could not find permutation for %s" %tuplex)
#-----


# Method for flattening and adding additional variables
def lepaugmentation(df, nlep, Truth=False):
    px  = awkward.fromiter(df['px'])
    py  = awkward.fromiter(df['py'])
    pz  = awkward.fromiter(df['pz'])
    E   = awkward.fromiter(df['E'])
    vtx = awkward.fromiter(df['vtxid'])
    pid = awkward.fromiter(df['pdgid'])


    # Make tlv - handy when computing angular variables
    tlv = uproot_methods.classes.TLorentzVector.TLorentzVectorArray.from_cartesian(px, py, pz, E)

    df["tlv"] = tlv[:]

    df["pt"]  = tlv[:].pt
    pt  = awkward.fromiter(df['pt'])
    pt_org  = awkward.fromiter(df['pt'])
    df["phi"] = tlv[:].phi
    phi = awkward.fromiter(df['phi'])
    
    df["theta"] = tlv.theta
    theta  = awkward.fromiter(df['theta'])

    df["eta"] = tlv.eta
    eta  = awkward.fromiter(df['eta'])
        
    px  = px.pad(nlep).fillna(-999)
    py  = py.pad(nlep).fillna(-999)
    pz  = pz.pad(nlep).fillna(-999)
    E   = E.pad(nlep).fillna(-999)
    pt  = pt.pad(nlep).fillna(-999)
    vtx = vtx.pad(nlep).fillna(-999)
    pid = pid.pad(nlep).fillna(-999)

    phi  = phi.pad(nlep).fillna(-999)
    eta  = eta.pad(nlep).fillna(-999)
    theta  = theta.pad(nlep).fillna(-999)

    print("Make vars:")
    # Make the lepton variables
    for i in range(1,nlep+1):
        df['lep%i_pt'%i]  = pt[pt.argmax()].flatten()
        df['lep%i_phi'%i]  = phi[pt.argmax()].flatten()
        df['lep%i_eta'%i]  = eta[pt.argmax()].flatten()
        df['lep%i_theta'%i]  = theta[pt.argmax()].flatten()
        df['lep%i_px'%i]  = px[pt.argmax()].flatten()
        df['lep%i_py'%i]  = py[pt.argmax()].flatten()
        df['lep%i_pz'%i]  = pz[pt.argmax()].flatten()
        df['lep%i_E' %i]  = E[pt.argmax()].flatten()
        df['lep%i_vtx'%i]   = vtx[pt.argmax()].flatten()
        df['lep%i_pid'%i]   = pid[pt.argmax()].flatten()
        df['lep%i_tlv'%i]   = tlv[pt.argmax()].flatten()

        mask = np.logical_and(pt != pt.max(), pt.max() != -999)
        
        for j in range(len(mask)):
            if sum(mask[j]) < 4-i:            
                print(mask[j], j)
                #print(df.evnum[j])
        
        px    =   px[mask]   
        py    =   py[mask]   
        pz    =   pz[mask]   
        E     =   E[mask]    
        pt    =   pt[mask]   
        vtx   =   vtx[mask]  
        pid   =   pid[mask]  
        phi   =   phi[mask]  
        eta   =   eta[mask]  
        theta =   theta[mask]
        tlv   =   tlv[mask]
        

    # Compute variables for all combinations of 2 leptons
    pairs = pt_org.argchoose(2)
    print("pairs:", pairs)
    left  = pairs.i0
    right = pairs.i1

    #print(df.head())
    #print(df["lep1_tlv"].head())

    print("left = ", left)
    print("right = ", right)
    
    for ilep in range(len(left[0])):
        i = left[0][ilep]
        j = right[0][ilep]
        print('i = %i, j = %i'%(i,j))
        idx1 = left[0][i]
        idx2 = right[0][i]

        df['mll_%i%i'%(i+1,j+1)]   = (df['lep%i_tlv'%(i+1)]+df['lep%i_tlv'%(j+1)]).apply(get_invmass)
        df['dphi_%i%i'%(i+1,j+1)] = df.apply(lambda x : get_deltaPhi(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1)
        df['dR_%i%i'%(i+1,j+1)]   = df.apply(lambda x : get_deltaR(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1)


    if Truth:
        df['target'] = df.apply(lambda x : classify_event(x['lep1_vtx'],x['lep2_vtx'],x['lep3_vtx'],x['lep4_vtx'],x['lep1_pid'],x['lep2_pid'],x['lep3_pid'],x['lep4_pid']), axis=1)

    
    df = df.drop(['px', 'py', 'pz', 'pt', 'E', 'vtxid', 'pdgid', 'evnum', 'onshell_w', 'tlv', 'phi', 'theta', 'eta', 'lep1_tlv', 'lep2_tlv', 'lep3_tlv', 'lep4_tlv'], axis=1)
    df = df.select_dtypes(exclude=['int32'])

    return df
#-----


def Make_DF(File="N1_50",  Period="", Truth=False, Save=False):
    """ 
    This function reads a ROOT-file, converts it to a dataframe and makes new variables to be added to a new dataframe. Can choose to save the file as .h5-file or not.
    Parameters:
        File - str; Name used to choose the file to convert. Same name used as the save-name.
        Period - str; Choose which period/folder for the backgrounds. Leave empty for signals.
        Truth - bool; When True, a target variable is made and added to the dataframe. Used for when the signals are to be used for training classification models later.
        Save - bool; When True, save the dataframe as .h5-file.
    """
    Folder = "/scratch2/Master_krilangs/Trilepton_Ntuples/Skimslim/"
    suffix = "_merged_processed"
    print("\u0332".join(File + " "))

    if Period == "18":
        SubFolder = "data18_mc16e/"
    elif Period == "17":
        SubFolder = "data17_mc16d/"
    elif Period == "1516":
        SubFolder = "data1516_mc16a/"
    else:
        SubFolder = ""

    
    if File == "N1_50":
        file50 = "../myfile_allevents"
        tree = uproot.open(Folder+file50+".root")["mytree"]
        df = tree.pandas.df(flatten = False)
        del(tree)   # Free up memory

    elif File == "N1_150":
        file150 = "myfile_VERTEX_LFC_150_200_250"
        tree = uproot.open(Folder + file150 + ".root")["mytree"]
        df = tree.pandas.df(flatten = False)
        del(tree)  # Free up memory
        df = df.drop([df.index[11071], df.index[23774], df.index[60373], df.index[40743]])

    elif File == "N1_450":
        file450 = "myfile_VERTEX_LFC_450_500_550"
        tree = uproot.open(Folder + file450 + ".root")["mytree"]
        df = tree.pandas.df(flatten = False)
        del(tree)  # Free up memory
        df = df.drop([df.index[14678], df.index[26355], df.index[39870], df.index[76527], df.index[60540], df.index[125862]])

    else:
        file = SubFolder + File + suffix
        tree = uproot.open(Folder + file + ".root")[File + "_NoSys"]
        df_tree = tree.pandas.df(flatten = False)
        del(tree)   # Free up memory
        df = df_tree.iloc[:,64:74]
        del(df_tree)  # Free up memory
        if File == "diboson3L":
            df = df.drop([df.index[178412]])
        if File == "diboson4L":
            df = df.drop([df.index[3826200]])
        if File == "topOther":
            df = df.drop([df.index[3281191]])
        if File == "ttbar":
            df = df.drop([df.index[5690459], df.index[3723599]])

    newdf = lepaugmentation(df, 4, Truth)
    del(df)  # Free up memory

    if Save:
        print("Save:")
        newdf.to_hdf("Trilepton_ML.h5", key=File+Period)  # Save dataframe to file.

    print(newdf.info(verbose=True))
    #print(newdf["target"].value_counts())
    
    del(newdf)  # Free up memory
#-----

"""
Available variables to plot (number of parameters in total in parenthesis):
    -File (13): N1_50, N1_150, N1_450, diboson2L, diboson3L, diboson4L, higgs, singletop, topOther, triboson, ttbar, Zjets, LFCMN1150, LFCMN1450
    -Period (4): "", 1516, 17, 18
"""

#Make_DF(File="N1_50",  Period="", Truth=True, Save=False)
#Make_DF(File="N1_150",  Period="", Truth=True, Save=True)
#Make_DF(File="N1_450",  Period="", Truth=True, Save=True)
#Make_DF(File="LFCMN1150", Period="", Truth=False, Save=True)
#Make_DF(File="LFCMN1450", Period="", Truth=False, Save=True)
#Make_DF(File="diboson3L",  Period="18", Truth=False, Save=True)
"""
bkgs = ["ttbar", "Zjets"]
for name in bkgs:
    Make_DF(File=name,  Period="18", Truth=False, Save=True)
"""
f = h5py.File("Trilepton_ML.h5", "r")
print([key for key in f.keys()])

