import uproot
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from math import sqrt, cos
import awkward
import matplotlib.pyplot as plt
import uproot_methods.classes.TLorentzVector
import sys
import itertools

perm = list(itertools.permutations([1, 2, 3]))
#print(perm)
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
        df['dR_%i%i'%(i+1,j+1)]   = df.apply(lambda x : get_deltaPhi(x['lep%i_tlv'%(i+1)],x['lep%i_tlv'%(j+1)]), axis=1)

    
    df["target"] = df.apply(lambda x : classify_event(x['lep1_vtx'],x['lep2_vtx'],x['lep3_vtx'],x['lep4_vtx'],x['lep1_pid'],x['lep2_pid'],x['lep3_pid'],x['lep4_pid']), axis=1)

    

    df = df.drop(['px', 'py', 'pz', 'pt', 'E', 'vtxid', 'pdgid', 'evnum','tlv', 'phi', 'theta', 'eta'], axis=1)
    return df

#Folder = "../ntuples/heavy_neutrinos/"    
Folder = "/scratch2/Master_krilangs/Trilepton_Ntuples/"

size = input("Choose Ntuple size (small/big):")  # May remove later
if size == "small":
    file = "myfile"
elif size == "big":
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
df1 = tree.pandas.df(branches=var)
df3 = tree.pandas.df(branches=var, flatten = False)

N1 = len(df1.nlep)
N2 = int(N1/4)

y = [1, 2, 3, 4]*(N2)
y = np.array(y)

df1["target"] = y
#print(df1.keys())


#df3 = df1.copy(True)

newdf = lepaugmentation(df3,4)

print(newdf.info())
print(newdf.head())

x = np.linspace(0,500,250)
for i in range(1,4):
    for j in range(i+1,4):
        if i == j: continue
        ax = (newdf['mll_%i%i'%(i,j)]/1000.).hist(bins=x)
        ax.legend()
plt.show()

#newdf.to_hdf("Trilepton_ML.h5", key="DF_flat2")
#df1.to_hdf("Trilepton_ML.h5", key = size+"_original")  # Save dataframe to file

sys.exit()

"""Make angular variables for each lepton. Very messy and long right now, as I am not sure which method to use yet."""
angles = ["phi", "eta"] #"phi_1", "phi_2", "phi_3", "phi_MET", "eta_1", "eta_2", "eta_3", "eta_MET"]
dPhi_list = ["dPhi_12", "dPhi_13", "dPhi_23", "dPhi_1MET", "dPhi_2MET", "dPhi_3MET"]
dEta_list = ["dEta_12", "dEta_13", "dEta_23", "dEta_1MET", "dEta_2MET", "dEta_3MET"]
dR_list = ["dR_12", "dR_13", "dR_23", "dR_1MET", "dR_2MET", "dR_3MET"]

### Used for alternative angular
dPhi_alt_list = ["dPhi_1", "dPhi_2", "dPhi_3"]
dEta_alt_list = ["dEta_1", "dEta_2", "dEta_3"]
dR_alt_list = ["dR_1", "dR_2", "dR_3"]

dPhi_alt = np.zeros((N1, len(dPhi_alt_list)))
dEta_alt = np.zeros((N1, len(dEta_alt_list)))
dR_alt = np.zeros((N1, len(dR_alt_list)))
###

phi_eta = np.zeros((N1, len(angles)))
dPhi = np.zeros((N1, len(dPhi_list)))
dEta = np.zeros((N1, len(dEta_list)))
dR = np.zeros((N1, len(dR_list)))


j = 0
print("Loop start")
# Loop that makes the angular variables to be implemented into a dataframe later:
for i in range(N2):
    if i%20000 == 0:
        print("%i/%i" %(i,N2))
    vec0 = TLorentzVector(px[i][0], py[i][0], pz[i][0], E[i][0])
    vec1 = TLorentzVector(px[i][1], py[i][1], pz[i][1], E[i][1])
    vec2 = TLorentzVector(px[i][2], py[i][2], pz[i][2], E[i][2])
    vec3 = TLorentzVector(px[i][3], py[i][3], pz[i][3], E[i][3])

    phi_eta[i+j][0] = vec0.Phi()    # Phi_1
    phi_eta[i+1+j][0] = vec1.Phi()  # Phi_2
    phi_eta[i+2+j][0] = vec2.Phi()  # Phi_3
    phi_eta[i+3+j][0] = vec3.Phi()  # Phi_MET
    phi_eta[i+j][1] = vec0.Eta()    # Eta_1
    phi_eta[i+1+j][1] = vec1.Eta()  # Eta_2
    phi_eta[i+2+j][1] = vec2.Eta()  # Eta_3
    phi_eta[i+3+j][1] = vec3.Eta()  # Eta_MET

    # Alternate angular with only three variables each lepton
    dPhi_alt[i+0+j][0] = abs(vec0.DeltaPhi(vec1))  # dPhi_12 at 1
    dPhi_alt[i+0+j][1] = abs(vec0.DeltaPhi(vec2))  # dPhi_13 at 1
    dPhi_alt[i+0+j][2] = abs(vec0.DeltaPhi(vec3))  # dPhi_1MET at 1
    dPhi_alt[i+1+j][0] = abs(vec1.DeltaPhi(vec0))  # dPhi_12 at 2
    dPhi_alt[i+1+j][1] = abs(vec1.DeltaPhi(vec2))  # dPhi_23 at 2
    dPhi_alt[i+1+j][2] = abs(vec1.DeltaPhi(vec3))  # dPhi_2MET at 2
    dPhi_alt[i+2+j][0] = abs(vec2.DeltaPhi(vec0))  # dPhi_13 at 3
    dPhi_alt[i+2+j][1] = abs(vec2.DeltaPhi(vec1))  # dPhi_23 at 3
    dPhi_alt[i+2+j][2] = abs(vec2.DeltaPhi(vec3))  # dPhi_3MET at 3
    dPhi_alt[i+3+j][0] = abs(vec3.DeltaPhi(vec0))  # dPhi_1MET at 4
    dPhi_alt[i+3+j][1] = abs(vec3.DeltaPhi(vec1))  # dPhi_2MET at 4
    dPhi_alt[i+3+j][2] = abs(vec3.DeltaPhi(vec2))  # dPhi_3MET at 4

    dEta_alt[i+0+j][0] = abs(phi_eta[i+j][1]-phi_eta[i+1+j][1])  # dEta_12 at 1
    dEta_alt[i+0+j][1] = abs(phi_eta[i+j][1]-phi_eta[i+2+j][1])  # dEta_13 at 1
    dEta_alt[i+0+j][2] = abs(phi_eta[i+j][1]-phi_eta[i+3+j][1])  # dEta_1MET at 1
    dEta_alt[i+1+j][0] = abs(phi_eta[i+1+j][1]-phi_eta[i+j][1])  # dEta_12 at 2
    dEta_alt[i+1+j][1] = abs(phi_eta[i+1+j][1]-phi_eta[i+2+j][1])  # dEta_23 at 2
    dEta_alt[i+1+j][2] = abs(phi_eta[i+1+j][1]-phi_eta[i+3+j][1])  # dEta_2MET at 2
    dEta_alt[i+2+j][0] = abs(phi_eta[i+2+j][1]-phi_eta[i+j][1])  # dEta_13 at 3
    dEta_alt[i+2+j][1] = abs(phi_eta[i+2+j][1]-phi_eta[i+1+j][1])  # dEta_23 at 3
    dEta_alt[i+2+j][2] = abs(phi_eta[i+2+j][1]-phi_eta[i+3+j][1])  # dEta_3MET at 3
    dEta_alt[i+3+j][0] = abs(phi_eta[i+3+j][1]-phi_eta[i+j][1])  # dEta_1MET at 4
    dEta_alt[i+3+j][1] = abs(phi_eta[i+3+j][1]-phi_eta[i+1+j][1])  # dEta_2MET at 4
    dEta_alt[i+3+j][2] = abs(phi_eta[i+3+j][1]-phi_eta[i+2+j][1])  # dEta_3MET at 4

    dR_alt[i+0+j][0] = abs(vec0.DeltaR(vec1))  # dR_12 at 1
    dR_alt[i+0+j][1] = abs(vec0.DeltaR(vec2))  # dR_13 at 1
    dR_alt[i+0+j][2] = abs(vec0.DeltaR(vec3))  # dR_1MET at 1
    dR_alt[i+1+j][0] = abs(vec1.DeltaR(vec0))  # dR_12 at 2
    dR_alt[i+1+j][1] = abs(vec1.DeltaR(vec2))  # dR_23 at 2
    dR_alt[i+1+j][2] = abs(vec1.DeltaR(vec3))  # dR_2MET at 2
    dR_alt[i+2+j][0] = abs(vec2.DeltaR(vec0))  # dR_13 at 3
    dR_alt[i+2+j][1] = abs(vec2.DeltaR(vec1))  # dR_23 at 3
    dR_alt[i+2+j][2] = abs(vec2.DeltaR(vec3))  # dR_3MET at 3
    dR_alt[i+3+j][0] = abs(vec3.DeltaR(vec0))  # dR_1MET at 4
    dR_alt[i+3+j][1] = abs(vec3.DeltaR(vec1))  # dR_2MET at 4
    dR_alt[i+3+j][2] = abs(vec3.DeltaR(vec2))  # dR_3MET at 4

    """
    #Absolute values where the relevant leptons have values and the other two don't
    dPhi[i+0+j][0] = abs(vec0.DeltaPhi(vec1))  # dPhi_12 at 1
    dPhi[i+1+j][0] = abs(vec0.DeltaPhi(vec1))  # dPhi_12 at 2
    dPhi[i+0+j][1] = abs(vec0.DeltaPhi(vec2))  # dPhi_13 at 1
    dPhi[i+2+j][1] = abs(vec0.DeltaPhi(vec2))  # dPhi_13 at 3
    dPhi[i+1+j][2] = abs(vec1.DeltaPhi(vec2))  # dPhi_23 at 2
    dPhi[i+2+j][2] = abs(vec1.DeltaPhi(vec2))  # dPhi_23 at 3
    dPhi[i+0+j][3] = abs(vec0.DeltaPhi(vec3))  # dPhi_1MET at 1
    dPhi[i+3+j][3] = abs(vec0.DeltaPhi(vec3))  # dPhi_1MET at 4
    dPhi[i+1+j][4] = abs(vec1.DeltaPhi(vec3))  # dPhi_2MET at 2
    dPhi[i+3+j][4] = abs(vec1.DeltaPhi(vec3))  # dPhi_2MET at 4
    dPhi[i+2+j][5] = abs(vec2.DeltaPhi(vec3))  # dPhi_3MET at 3
    dPhi[i+3+j][5] = abs(vec2.DeltaPhi(vec3))  # dPhi_3MET at 4
    
    dEta[i+0+j][0] = abs(phi_eta[i+j][1]-phi_eta[i+1+j][1])    # dEta_12 at 1
    dEta[i+1+j][0] = abs(phi_eta[i+j][1]-phi_eta[i+1+j][1])    # dEta_12 at 2
    dEta[i+0+j][1] = abs(phi_eta[i+j][1]-phi_eta[i+2+j][1])    # dEta_13 at 1
    dEta[i+2+j][1] = abs(phi_eta[i+j][1]-phi_eta[i+2+j][1])    # dEta_13 at 3
    dEta[i+1+j][2] = abs(phi_eta[i+1+j][1]-phi_eta[i+2+j][1])  # dEta_23 at 2
    dEta[i+2+j][2] = abs(phi_eta[i+1+j][1]-phi_eta[i+2+j][1])  # dEta_23 at 3
    dEta[i+0+j][3] = abs(phi_eta[i+j][1]-phi_eta[i+3+j][1])    # dEta_1MET at 1
    dEta[i+3+j][3] = abs(phi_eta[i+j][1]-phi_eta[i+3+j][1])    # dEta_1MET at 4
    dEta[i+1+j][4] = abs(phi_eta[i+1+j][1]-phi_eta[i+3+j][1])  # dEta_2MET at 2
    dEta[i+3+j][4] = abs(phi_eta[i+1+j][1]-phi_eta[i+3+j][1])  # dEta_2MET at 4
    dEta[i+2+j][5] = abs(phi_eta[i+2+j][1]-phi_eta[i+3+j][1])  # dEta_3MET at 3
    dEta[i+3+j][5] = abs(phi_eta[i+2+j][1]-phi_eta[i+3+j][1])  # dEta_3MET at 4
    dR[i+0+j][0] = abs(vec0.DeltaR(vec1))    # dR_12 at 1
    dR[i+1+j][0] = abs(vec0.DeltaR(vec1))    # dR_12 at 2
    dR[i+0+j][1] = abs(vec0.DeltaR(vec2))    # dR_13 at 1
    dR[i+2+j][1] = abs(vec0.DeltaR(vec2))    # dR_13 at 3
    dR[i+1+j][2] = abs(vec1.DeltaR(vec2))    # dR_23 at 2
    dR[i+2+j][2] = abs(vec1.DeltaR(vec2))    # dR_23 at 3
    dR[i+0+j][3] = abs(vec0.DeltaR(vec3))    # dR_1MET at 1
    dR[i+3+j][3] = abs(vec0.DeltaR(vec3))    # dR_1MET at 4
    dR[i+1+j][4] = abs(vec1.DeltaR(vec3))    # dR_2MET at 2
    dR[i+3+j][4] = abs(vec1.DeltaR(vec3))    # dR_2MET at 4
    dR[i+2+j][5] = abs(vec2.DeltaR(vec3))    # dR_3MET at 3
    dR[i+3+j][5] = abs(vec2.DeltaR(vec3))    # dR_3MET at 4
    """
    """
    for k in range(4):
        # All leptons have the same angular values for each event and variable
        dPhi[i+k+j][0] = vec0.DeltaPhi(vec1)  # dPhi_12
        dPhi[i+k+j][1] = vec0.DeltaPhi(vec2)  # dPhi_13
        dPhi[i+k+j][2] = vec1.DeltaPhi(vec2)  # dPhi_23
        dPhi[i+k+j][3] = vec0.DeltaPhi(vec3)  # dPhi_1MET
        dPhi[i+k+j][4] = vec1.DeltaPhi(vec3)  # dPhi_2MET
        dPhi[i+k+j][5] = vec2.DeltaPhi(vec3)  # dPhi_3MET
        dEta[i+k+j][0] = phi_eta[i+j][1]-phi_eta[i+1+j][1]    # dEta_12
        dEta[i+k+j][1] = phi_eta[i+j][1]-phi_eta[i+2+j][1]    # dEta_13
        dEta[i+k+j][2] = phi_eta[i+1+j][1]-phi_eta[i+2+j][1]  # dEta_23
        dEta[i+k+j][3] = phi_eta[i+j][1]-phi_eta[i+3+j][1]    # dEta_1MET
        dEta[i+k+j][4] = phi_eta[i+1+j][1]-phi_eta[i+3+j][1]  # dEta_2MET
        dEta[i+k+j][5] = phi_eta[i+2+j][1]-phi_eta[i+3+j][1]  # dEta_3MET
        dR[i+k+j][0] = vec0.DeltaR(vec1)    # dR_12
        dR[i+k+j][1] = vec0.DeltaR(vec2)    # dR_13
        dR[i+k+j][2] = vec1.DeltaR(vec2)    # dR_23
        dR[i+k+j][3] = vec0.DeltaR(vec3)    # dR_1MET
        dR[i+k+j][4] = vec1.DeltaR(vec3)    # dR_2MET
        dR[i+k+j][5] = vec2.DeltaR(vec3)    # dR_3MET
        
        #Absolute values
        dPhi[i+k+j][0] = abs(vec0.DeltaPhi(vec1))  # dPhi_12
        dPhi[i+k+j][1] = abs(vec0.DeltaPhi(vec2))  # dPhi_13
        dPhi[i+k+j][2] = abs(vec1.DeltaPhi(vec2))  # dPhi_23
        dPhi[i+k+j][3] = abs(vec0.DeltaPhi(vec3))  # dPhi_1MET
        dPhi[i+k+j][4] = abs(vec1.DeltaPhi(vec3))  # dPhi_2MET
        dPhi[i+k+j][5] = abs(vec2.DeltaPhi(vec3))  # dPhi_3MET
        dEta[i+k+j][0] = abs(phi_eta[i+j][1]-phi_eta[i+1+j][1])    # dEta_12
        dEta[i+k+j][1] = abs(phi_eta[i+j][1]-phi_eta[i+2+j][1])    # dEta_13
        dEta[i+k+j][2] = abs(phi_eta[i+1+j][1]-phi_eta[i+2+j][1])  # dEta_23
        dEta[i+k+j][3] = abs(phi_eta[i+j][1]-phi_eta[i+3+j][1])    # dEta_1MET
        dEta[i+k+j][4] = abs(phi_eta[i+1+j][1]-phi_eta[i+3+j][1])  # dEta_2MET
        dEta[i+k+j][5] = abs(phi_eta[i+2+j][1]-phi_eta[i+3+j][1])  # dEta_3MET
        dR[i+k+j][0] = abs(vec0.DeltaR(vec1))    # dR_12
        dR[i+k+j][1] = abs(vec0.DeltaR(vec2))    # dR_13
        dR[i+k+j][2] = abs(vec1.DeltaR(vec2))    # dR_23
        dR[i+k+j][3] = abs(vec0.DeltaR(vec3))    # dR_1MET
        dR[i+k+j][4] = abs(vec1.DeltaR(vec3))    # dR_2MET
        dR[i+k+j][5] = abs(vec2.DeltaR(vec3))    # dR_3MET
    """
    j += 3
print("Loop finished")




"""Add angular variables to dataframes and save the dataframes for later."""
print("Add angular")
"""
# Test a variant with all angular variables
df2 = pd.DataFrame()
df2["pt"] = df1.pt
for i in range(len(angles)):
    df2[angles[i]] = phi_eta[:,i]
for j in range(len(dPhi_alt_list)):
    df2[dPhi_alt_list[j]] = dPhi_alt[:,j]
for k in range(len(dEta_alt_list)):
    df2[dEta_alt_list[k]] = dEta_alt[:,k]
for l in range(len(dR_alt_list)):
    df2[dR_alt_list[l]] = dR_alt[:,l]
df2["target"] = df1.target
print(df2.info())
print(df2.head())
df2.to_hdf("Trilepton_ML.h5", key = size+"_alt_angular")
"""
"""
# Full DF with all angular variables
#leftindex = pd.MultiIndex.from_product([range(N2), [0,1,2,3]], names=["entry", "subentry"])
df2 = pd.DataFrame()#index=leftindex)
df2["pt"] = df1.pt.values
for i in range(len(angles)):
    df2[angles[i]] = phi_eta[:,i]
for j in range(len(dPhi_list)):
    df2[dPhi_list[j]] = dPhi[:,j]
for k in range(len(dEta_list)):
    df2[dEta_list[k]] = dEta[:,k]
for l in range(len(dR_list)):
    df2[dR_list[l]] = dR[:,l]
df2["target"] = df1.target.values
print(len(df2))
#print(df2)#.loc[:, "dPhi_12":"dPhi_3MET"])
#df2.to_hdf("Trilepton_ML.h5", key = size+"_angular_fullevents")
"""

"""
# Basic DF with pt, phi and eta variables
new_df = pd.DataFrame(index=leftindex)
new_df["pt"] = df1.pt
for i in range(len(angles)):
    new_df[angles[i]] = phi_eta[:,i]
new_df["target"] = df1.target
#print(new_df)
#new_df.to_hdf("Trilepton_ML.h5", key = "pt_phi_eta")
"""


"""Calculate invariant mass of N."""
"""
print("Invariant mass:")
m_N = np.zeros(N2)
#M_N = np.zeros(N)
#m_pt = np.zeros(N2)
#W = np.zeros(N2)
for i in range(N2):
    Energy_sum = E[i][2] + E[i][3] + E[i][1]
    Mom_norm = px[i][2]**2 + py[i][2]**2 + pz[i][2]**2 + px[i][3]**2 + py[i][3]**2 + pz[i][3]**2 + px[i][1]**2 + py[i][1]**2 + pz[i][1]**2
    #if Mom_norm > Energy_sum:
    #    print(i)
    #    print("Energy:",Energy_sum, "Mom:",Mom_norm)
    m_N[i] = sqrt((Energy_sum)**2 - Mom_norm)
    #m_pt[i] = sqrt((pt[i][1]+pt[i][2]+pt[i][3])**2)
    #W[i] = sqrt(Energy_sum**2 - (pt[i][1]+pt[i][2]+pt[i][3])**2 - (pz[i][1]+pz[i][2]+pz[i][3])**2)
    #M_N[i] = sqrt( 2*pt[i][2]*pt[i][3]*pt[i][1]*(np.cosh(eta[i][2]-eta[i][3]-eta[i][1])-np.cos(phi[i][2]-phi[i][3]-phi[i][1])) )
#print(m_N[:6])
print("Mean mass sum:", np.mean(m_N))
#print(np.mean(m_pt))
#print(np.mean(W))
#print("Mass collider", M_N[:4])
"""
