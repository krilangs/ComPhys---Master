import sys
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
nupid = [-12,12,-14,14,-16, 16]

def get_mt(x,y):
    #print("Type of value is ",type(value))
    try:
        dphi = x.delta_phi(y)
        print(dphi)
        mtw = sqrt(2*x.pt()*y.pt()*(1-cos(dphi)))
        print(mtw)
        return(mtw)
    except:
        print("Could not return transverse mass for ",(x,y))
        return -1

def get_invmass(value):
    #print("Type of value is ",type(value))
    try:
        return(value.mass)
    except:
        print("Could not return mass for ",value)
        return -1


def get_deltaPhi(x,y):
    #print("Type of value is ",type(x))
    try:
        return(x.delta_phi(y))
    except:
        print("Could not return deltaPhi for ",(x,y))
        return -1


def get_deltaR(x,y):
    #print(" of value is ",type(value))
    try:
        return(x.delta_r(y))
    except:
        print("Could not return deltaR for ",(x,y))
        return -1


def classify_event(l1,l2,l3,l4,p1,p2,p3,p4):
    if p1 in nupid: tuplex = l2,l3,l4
    elif p2 in nupid: tuplex = l1,l3,l4
    elif p3 in nupid: tuplex = l1,l2,l4
    elif p4 in nupid: tuplex = l1,l2,l3
    if tuplex in perm:
        return perm[perm.index(tuplex)]
    else:
        print("Could not find permutation for %s"%tuplex)

# Method for flattening and adding additional variables
def lepaugmentation(df,nlep):
    
    px  = awkward.fromiter(df['px'])
    py  = awkward.fromiter(df['py'])
    pz  = awkward.fromiter(df['pz'])
    E   = awkward.fromiter(df['E'])
    vtx = awkward.fromiter(df['vtxid'])
    pid = awkward.fromiter(df['pdgid'])


    # make tlv - handy when computing angular variables
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
    pt   = pt.pad(nlep).fillna(-999)
    vtx = vtx.pad(nlep).fillna(-999)
    pid = pid.pad(nlep).fillna(-999)

    phi  = phi.pad(nlep).fillna(-999)
    eta  = eta.pad(nlep).fillna(-999)
    theta  = theta.pad(nlep).fillna(-999)

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

        mask = np.logical_and(pt != pt.max(), pt.max != -999)
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

    print(df.head())
    print(df["lep1_tlv"].head())

    #f = lambda x,y : x.delta_phi(y)

    print("left = ",left)
    print("right = ",right)
    
    for ilep in range(len(left[0])):
        i = left[0][ilep]
        j = right[0][ilep]
        #for j in right[0]:
        print("i = %i, j = %i"%(i,j))
        idx1 = left[0][i]
        idx2 = right[0][i]
        df['mll_%i%i'%(i+1,j+1)]   = (df["lep%i_tlv"%(i+1)]+df["lep%i_tlv"%(j+1)]).apply(get_invmass)
        #df['mt_%i%i'%(i+1,j+1)]   = df.apply(lambda x : get_mt(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1) # not yet working
        df['dphi_%i%i'%(i+1,j+1)] = df.apply(lambda x : get_deltaPhi(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1)
        df['dR_%i%i'%(i+1,j+1)]   = df.apply(lambda x : get_deltaR(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1)

    
    df["target"] = df.apply(lambda x : classify_event(x['lep1_vtx'],x['lep2_vtx'],x['lep3_vtx'],x['lep4_vtx'],x['lep1_pid'],x['lep2_pid'],x['lep3_pid'],x['lep4_pid']), axis=1)

    
    df = df.drop(['px', 'py', 'pz', 'pt', 'E', 'vtxid', 'pdgid', 'evnum','tlv', 'phi', 'theta', 'eta'], axis=1)
    return df


#Folder = "../ntuples/heavy_neutrinos/"    
Folder = "/scratch2/Master_krilangs/Trilepton_Ntuples/"

#size = input("Choose Ntuple size (small/big):")  # May remove later
#if size == "small":
#    file = "myfile"
#elif size == "big":
file = "myfile_allevents"


"""Read from ROOT."""
tree = uproot.open(Folder+file+".root")["mytree"]

var = ["nlep", "px", "py", "pz", "pt", "E", "vtxid", "pdgid", "evnum", "onshell_w"]

for key, name in zip(tree.keys(), var):
    globals()[name] = tree.array(key)

#tree.show()
#print(tree.keys())
#print(tree.keys()[0])


"""Make a dataframe from the ROOT-file, make the targets and add them to the dataframe."""
df = tree.pandas.df(branches=var, flatten = False)

newdf = lepaugmentation(df,4)

print(newdf.info())
print(newdf.head())

x = np.linspace(0,500,250)
for i in range(1,4):
    for j in range(i+1,4):
        if i == j: continue
        ax = (newdf['mll_%i%i'%(i,j)]/1000.).hist(bins=x)
        ax.legend()
plt.show()

#newdf.to_hdf("Trilepton_ML.h5", key="DF_flat3")
del(newdf)  # Free up memory
