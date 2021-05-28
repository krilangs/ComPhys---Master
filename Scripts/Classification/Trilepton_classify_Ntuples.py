import sys
import h5py
import uproot
import awkward
import numpy as np
import pandas as pd
import pickle as pkl
import uproot_methods.classes.TLorentzVector


def Classify_Ntuples(File="", Signal="", data18=False):
    """
    Function that converts dataframes of Ntuples to .csv-files, which can later be converted to ROOT.
    -File: str, Name of the file.
    -Signal: str, Which signal to use, 150 or 450.
    -data18, bool, When True, load background ntuple from a folder. When False, load a signal ntuple.
    """
    print("Ntuple %s and signal %s" %(File, Signal))
    Folder = "/scratch2/Master_krilangs/Trilepton_Ntuples/Skimslim/"
    suffix = "_merged_processed"
    if data18:
        file = "data18_mc16e/" + File + suffix
    else:
        file = File + suffix

    tree = uproot.open(Folder + file + ".root")[File + "_NoSys"]
    df_tree = tree.pandas.df(flatten = False)
    del(tree)  # Free up memory

    if data18:
        df_vars = pd.read_hdf("Trilepton_ML.h5", key=File + "18")
    else:
        df_vars = pd.read_hdf("Trilepton_ML.h5", key=File)

    merged = pd.concat([df_tree, df_vars], axis=1)
    del(df_tree); del(df_vars)   # Free up memory

    merged = merged.select_dtypes(exclude=["object"])
    merged.dropna(inplace=True)
    merged = merged.reset_index(drop=True)

    # Make new df with only relevant variables for classification.
    Xtest = merged.iloc[:,61:]
    Xtest = Xtest.drop(["lep1_eta", "lep2_eta", "lep3_eta", "lep4_eta"], axis=1)

    # Load classification model to predict classes.
    with open("finalized_model_" + Signal + ".pkl", "rb") as file:
        model = pkl.load(file)
        merged["pred_class"] = model.predict(Xtest)

        print("Predicted counts:")
        print(merged.pred_class.value_counts())
        file.close()

    del(Xtest)  # Free up memory
    #print(merged.info())

    # Save dataframe as .csv-file to be converted to ROOT later.
    print("Convert to CSV")
    cols = merged.columns.to_numpy()
    Dtypes = merged.dtypes.to_numpy()

    # Convert types to be convertable with ROOT later.
    for i in range(len(Dtypes)):
        if Dtypes[i] == "bool":
            Dtypes[i] = "O"
        if Dtypes[i] == "float64":
            Dtypes[i] = "D"
        if Dtypes[i] == "float32":
            Dtypes[i] = "F"
        if Dtypes[i] == "int64":
            Dtypes[i] = "L"
        if Dtypes[i] == "int32":
            Dtypes[i] = "I"

    for btype,key in zip(Dtypes,cols):
        merged = merged.rename(columns={str(key):str(key)+"/"+str(btype)}, inplace=False)
        #print(merged.info(verbose=True))

    # Save as .csv.
    #if data18:
    merged.to_csv(File + "_" + Signal + "_classif.csv", index=False)
    #else:
        #merged.to_csv(File + "_classif.csv", index=False)

    del(merged)# Free up memory
#-----


# Check keys in file.
#f = h5py.File("Trilepton_ML.h5", "r")
#print([key for key in f.keys()])


"""
signals = ["150", "450"]
backgrounds = ["diboson2L", "diboson3L", "diboson4L", "higgs", "singletop", "topOther", "triboson", "ttbar", "Zjets"]

for sig in signals:
    for bkg in backgrounds:
        Classify_Ntuples(bkg, sig, True)
"""

""" For the signals located at another folder."""
Classify_Ntuples("LFCMN1150", "450", False)
Classify_Ntuples("LFCMN1450", "150", False)

