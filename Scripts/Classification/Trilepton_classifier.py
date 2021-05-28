import sys
import h5py
import warnings

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import xgboost as xgb
import scikitplot as skplt
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Tools
from collections import OrderedDict
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_hist_gradient_boosting  # Needed when importing the HistGradientBoostingClassifier, experimental feature
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix, classification_report, precision_score, cohen_kappa_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Models
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")#, category=UserWarning)
plt.rcParams.update({'font.size': 12})


def print_keys(HDF5):
    """ 
    Function for printing the keys in a HDF5-file.  
    """
    print("Check keys in HDF5-file:")
    f = h5py.File(HDF5+".h5", "r")
    print([key for key in f.keys()])
    sys.exit()

#print_keys("Trilepton_ML")


"""Read dataframe(s) from file."""
df_N1_150 = pd.read_hdf("Trilepton_ML.h5", key="N1_150")
df_N1_450 = pd.read_hdf("Trilepton_ML.h5", key="N1_450")


def Select_DF(df, Simplify=False):
    """
    Function for dropping unnecessary fatures and drop rows with NaN values (if they exist).
    """
    df = df.select_dtypes(exclude=["int32"])
    #df.isnull() # Returns a boolean matrix, if the value is NaN then True otherwise False.
    #df.isnull().sum()   # Returns the column names along with the number of NaN values in that particular column.
    df.dropna(inplace=True)  # Removes rows in the dataframe containing NaN values.

    if Simplify: # Drop more variabels found unnecessary from analysis.
        df_simple = df.drop(["lep1_eta", "lep2_eta", "lep3_eta", "lep4_eta"], axis=1)
        return df_simple
    else:
        return df


"""Make design matrix X and target y from chosen dataframe."""
#features = Select_DF(df_N1_150, True)
features = Select_DF(df_N1_450, False)

X = features.select_dtypes(exclude=["object"])
features_names = list(X.columns)
Y = features.target.to_numpy()


# Change the targets from tuples to integer values
Target = np.zeros(len(Y))
for i in range(len(Y)):
    if Y[i] == (3,1,2):
        Target[i] = 312
    elif Y[i] == (1,3,2):
        Target[i] = 132
    elif Y[i] == (1,2,3):
        Target[i] = 123
    elif Y[i] == (3,2,1):
        Target[i] = 321
    elif Y[i] == (2,1,3):
        Target[i] = 213
    elif Y[i] == (2,3,1):
        Target[i] = 231
    else:
        print(Y[i])
        raise ValueError

y = pd.DataFrame({"target": Target}, dtype="int32")


"""Inspect the data."""
def data_info(df):
    df.info()
    print(df.head())
    print("Target counts:")
    print(y.target.value_counts())  # Print the counts of the different classes
    
    # Look at the correlations between the features
    print("Corr matrix:")
    plt.figure(figsize=(30,30))
    heatmap = sns.heatmap(df.corr(), annot=True, fmt=".1f", vmin=-1, vmax=1, center=0, linewidths=0.1, annot_kws={"size":8}, cbar=False)
    heatmap.set_title("Correlation heatmap: N1=450")

    # Find strong correlation pairs, drop diagonal
    corr_mat_pairs = df.corr().unstack()
    sort_corr = corr_mat_pairs.sort_values(kind="quicksort")
    strong_corrs = sort_corr[abs(sort_corr) > 0.7]
    strong_corrs = strong_corrs[abs(strong_corrs) < 1.0].drop_duplicates()
    print(strong_corrs)
    
    # Information gain of the features
    cols = list(X.columns)
    infos = mutual_info_classif(X.values, np.ravel(y))
    info_gain = {}
    for i in range(len(cols)):
        info_gain[str(cols[i])] = infos[i]
    
    info_gain = OrderedDict(sorted(info_gain.items(), key=lambda t: t[1]))
    print("Information gain of the features:")
    print(info_gain)
    
    plt.show()
    

"""Resample the data to make the datasets more balanced."""
def Resample(X, y, under=False, over=False):
    if under == True:
        print("Undersample")
        undersample = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        X, y = undersample.fit_resample(X, y)

    if over == True:
        print("Oversample")
        oversample = ADASYN(sampling_strategy="not majority", random_state=42)
        X, y = oversample.fit_resample(X, y)
    
    print(y.target.value_counts())  # Print the counts of the different classes after resampling
    return X, y 


"""Scale the data when called."""
def scaler(X_train, X_val, X_test):
    print("Scaling")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    return X_train, X_val, X_test


data_info(features)  # Call to inspect data
sys.exit()
X, y = Resample(X, y, under=False, over=True)  # Call to resample data


"""Split events into training, validation and test sets."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)


X_train, X_val, X_test = scaler(X_train, X_val, X_test)  # Call function for scaling the data.
#print(len(X_train), len(X_val), len(X_test)) 


"""Multiclass classification to identify leptons."""
def getTrainScores(gs):
    # Function that prints the RandomizedSearchCV best parameters and mean scores
    print("Start getTrainScores:")
    gs.fit(X_train, np.ravel(y_train))
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    print(results)
    print(best)
    return results, best


#Train models with various hyperparameters for optimization
model_LogRegCV = LogisticRegressionCV(multi_class="multinomial", random_state=42, n_jobs=-1, max_iter=1000)

model_DTC = DecisionTreeClassifier(random_state=42, max_features="auto", class_weight="balanced", max_depth=18)

model_ADC = AdaBoostClassifier(base_estimator=model_DTC, n_estimators=200, algorithm="SAMME.R", random_state=42)

model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=500, reg_alpha=10, learning_rate=0.2, subsample=0.9, colsample_bytree=0.85, min_child_weight=3, max_depth=10)

model_LGBM = LGBMClassifier(objective="multiclass", num_class=6, n_estimators=1600, max_depth=15, learning_rate=0.1, random_state=42, metric=["multi_logloss", "multi_error"], importance_type="gain")

model_MLP = MLPClassifier(alpha=0.1, max_iter=300, activation="relu", hidden_layer_sizes=(200,200,100,100,50), random_state=42)

model_RF = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, min_samples_split=5, max_depth=15, max_features="auto", random_state=42)

model_HistGBC = HistGradientBoostingClassifier(random_state=42, max_iter=300, max_depth=5, loss="categorical_crossentropy")

model_OvR = OneVsRestClassifier(model_RF)

model_OvO = OneVsOneClassifier(model_RF)

# Uncomment to use randomized search function to find best parameters of chosen model
#params={"max_depth":[-1,8,12,15,20], "learning_rate":[0.1,0.05,0.01, 0.005], "subsample":[1.0,0.9,0.95]}
#clf_LGBM = RandomizedSearchCV(estimator=model_LGBM, param_distributions=params,cv=3,n_jobs=-1,scoring="accuracy", random_state=42)
#getTrainScores(clf_LGBM)


"""Evaluate models with validation set."""
def eval_val(model, title):
    print("Start eval of model "+title+":")
    if title == "XGBoost" or title == "LGBM":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train, y_train)

    pred = model.predict(X_val)
    pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_val, pred)
    balanced_accuracy = balanced_accuracy_score(y_val, pred)
    variance = np.mean(np.var(pred))
    bias = np.mean((y_val.values - np.mean(pred))**2)

    if title == "XGBoost":
        # Plot logloss and error as function of iterations for train and test:
        print("Plot mlogloss and merror:")
        predictions = [round(value) for value in pred]
        new_acc = accuracy_score(y_val, predictions)
        results = model.evals_result()
        epochs = len(results["validation_0"]["merror"])
        x_axis = range(0,epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
        ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
        ax.legend()
        plt.title("Mlogloss plot")
        plt.ylabel("Log loss")
        plt.xlabel("Iterations")
        plt.tight_layout()
        fig,ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
        ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
        ax.legend()
        plt.title("Merror plot")
        plt.ylabel("Error")
        plt.xlabel("Iterations")
        plt.tight_layout()

    if title == "LGBM":
        # Plot logloss and error as function of iterations for train and test:
        print("Plot mlogloss and merror:")
        predictions = [round(value) for value in pred]
        new_acc = accuracy_score(y_val, predictions)
        results = model.evals_result_
        epochs = len(results["training"]["multi_error"])
        x_axis = range(0,epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results["training"]["multi_logloss"], label="Train")
        ax.plot(x_axis, results["valid_1"]["multi_logloss"], label="Test")
        ax.legend()
        plt.title("Multi_logloss plot")
        plt.ylabel("Log loss")
        plt.xlabel("Iterations")
        plt.tight_layout()
        fig,ax = plt.subplots()
        ax.plot(x_axis, results["training"]["multi_error"], label="Train")
        ax.plot(x_axis, results["valid_1"]["multi_error"], label="Test")
        ax.legend()
        plt.title("Multi_error plot")
        plt.ylabel("Error")
        plt.xlabel("Iterations")
        plt.tight_layout()
        
    fig, ax = plt.subplots(figsize=(8,8))
    model_plt = plot_confusion_matrix(model, X_val, y_val, normalize="true", ax=ax)
    model_plt.ax_.set_title("Confusion matrix: " + title)

    eval_val = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train], "BAcc":[balanced_accuracy], "Var":[round(variance,4)], "Bias":[round(bias,4)]}
    return pd.DataFrame(eval_val)



"""Select best model and assess results with test set."""
def eval_test(model, title):
    print("Assess final best " +title+ " model evaluation with test set:")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)
    print("Best iter:", model.best_iteration_)
    print("Best score:", model.best_score_)
    
    pred = model.predict(X_test)
    pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_test, pred)
    balanced_accuracy = balanced_accuracy_score(y_test, pred)
    variance = np.mean(np.var(pred))
    bias = np.mean((y_test.values - np.mean(pred))**2)
    prob = model.predict_proba(X_test)
    prob_train = model.predict_proba(X_train)
    logloss = log_loss(y_test, prob)

    cks = cohen_kappa_score(y_test, pred)
    cr = classification_report(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    print("Classification report "+title+": \n", cr)
    
    # Plot mlogloss and merror as function of iterations for train and test:
    print("Plot mlogloss and merror:")
    predictions = [round(value) for value in pred]
    new_acc = accuracy_score(y_val, predictions)
    results = model.evals_result_
    epochs = len(results["training"]["multi_error"])
    x_axis = range(0,epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results["training"]["multi_logloss"], label="Train")
    ax.plot(x_axis, results["valid_1"]["multi_logloss"], label="Test")
    ax.legend()
    plt.title("Multi_logloss plot")
    plt.ylabel("Log loss")
    plt.xlabel("Iterations")
    plt.tight_layout()
    fig,ax = plt.subplots()
    ax.plot(x_axis, results["training"]["multi_error"], label="Train")
    ax.plot(x_axis, results["valid_1"]["multi_error"], label="Test")
    ax.legend()
    plt.title("Multi_error plot")
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.tight_layout()

    
    print("Conf matrix:")
    print(conf_mat)
    fig, ax = plt.subplots(figsize=(8,8))
    plt_conf = plot_confusion_matrix(model, X_test, y_test, normalize="true", ax=ax)
    plt_conf.ax_.set_title("Confusion matrix best " + title + "model")

    print("Importance as Series and sort:")
    plt.figure(figsize=(8,8))
    importances = pd.Series(data=model.feature_importances_, index=features_names)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind="barh") #[-20:]
    plt.tight_layout()
    
    print("Importance with skplt:")
    skplt.estimators.plot_feature_importances(model, feature_names=features_names, title="Feature importance " + title, x_tick_rotation=90, figsize=(8,8))#, max_num_features=20)
    plt.tight_layout()
    
    
    print("ROC:")
    skplt.metrics.plot_roc(y_test, prob)

    print("Precision-Recall:")
    skplt.metrics.plot_precision_recall(y_test, prob)
    
    
    # Save best model
    """
    filename = "finalized_model_450.pkl"
    with open(filename, "wb") as file:
        pkl.dump(model, file)
    """

    eval_test = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train],"CKS":[cks], "BAcc":[balanced_accuracy], "LogLoss":[logloss], "Var":[round(variance,4)], "Bias":[round(bias,4)]}
    eval_test = pd.DataFrame(eval_test)
    print(eval_test.iloc[0,:])
    
    return eval_test



# Validation set:
"""
LogRegCV_DF = eval_val(model_LogRegCV, "LogRegCV")
DTC_DF = eval_val(model_DTC, "DecisionTree")
ADC_DF = eval_val(model_ADC, "AdaBoost")
RF_DF = eval_val(model_RF, "RandomForest")
OvR_DF = eval_val(model_OvR, "OvR")
OvO_DF = eval_val(model_OvO, "OvO")
MLP_DF = eval_val(model_MLP, "MLP")
HistGBC_DF = eval_val(model_HistGBC, "HGBC")

XGB_DF = eval_val(model_XGB, "XGBoost")
LGBM_DF = eval_val(model_LGBM, "LGBM")

#merge = pd.concat([LogRegCV_DF, DTC_DF, ADC_DF, RF_DF, OvR_DF, OvO_DF, MLP_DF, HistGBC_DF, XGB_DF, LGBM_DF])
merge = pd.concat([XGB_DF, LGBM_DF])
print(merge)
#print(LGBM_DF)
plt.show()
"""

# Test set:
best_model = eval_test(model_LGBM, "LGBM")
#print(best_model)


plt.show()

