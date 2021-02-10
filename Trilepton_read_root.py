import uproot
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from math import sqrt


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
N1 = len(df1.nlep)
N2 = int(N1/4)

y = [1, 2, 3, 4]*(N2)
y = np.array(y)
    
df1["target"] = y
print(df1.keys())

#df1.to_hdf("Trilepton_ML.h5", key = size+"_original")  # Save dataframe to file


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
