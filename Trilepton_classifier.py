import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, precision_score, cohen_kappa_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")#, category=UserWarning)

"""Read dataframe(s) from file."""
#df_Truth = pd.read_hdf("Trilepton_ML.h5", key="big_original")
#df_Angular = pd.read_hdf("Trilepton_ML.h5", key="big_angular")
#df_Angular_alt = pd.read_hdf("Trilepton_ML.h5", key="big_alt_angular")
#df_Angular_zeros = pd.read_hdf("Trilepton_ML.h5", key="big_angular_zeros")
#new_df = pd.DataFrame([df_Truth.pt, df_Angular.phi, df_Angular.eta, df_Truth.target]).transpose()
#df_short = pd.read_hdf("Trilepton_ML.h5", key="pt_phi_eta")
#df_Angular_fullevents = pd.read_hdf("Trilepton_ML.h5", key="big_angular_fullevents")

#df_flat = pd.read_hdf("Trilepton_ML.h5", key="DF_flat2")
df_flat = pd.read_hdf("Trilepton_ML.h5", key="DF_flat3")
df_flat = df_flat.select_dtypes(exclude=["int32"])

df_simple = df_flat.select_dtypes(exclude=["int32", "float64"])
df_simple = df_simple.drop(["lep1_theta", "lep1_px", "lep1_py", "lep1_pz", "lep1_E", "lep2_theta", "lep2_px", "lep2_py", "lep2_pz", "lep2_E", "lep3_theta", "lep3_px", "lep3_py", "lep3_pz", "lep3_E", "lep4_theta", "lep4_px", "lep4_py", "lep4_pz", "lep4_E"], axis=1)

#print(df_flat.info())
#print(df_flat.head())
#print(df_simple.info())


"""Make design matrix X and target y from dataframe."""
features = df_flat

X = features.select_dtypes(exclude=["object"])
features_names = list(X.columns)
Y = features.target

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
#print(y.target.value_counts())




"""Resample the data to make the datasets more balanced."""
def Resample(X, y, scale=False, under=False, over=False):
    if scale == True:
        print("Scale")
        X = StandardScaler().fit_transform(X)

    if under == True:
        print("Undersample")
        undersample = RandomUnderSampler(sampling_strategy="majority")#, random_state=42)
        X, y = undersample.fit_resample(X, y)

    if over == True:
        print("Oversample")
        oversample = ADASYN(sampling_strategy="not majority")#,random_state=42)
        X, y = oversample.fit_resample(X, y)
    
    print(y.target.value_counts())
    
    return X, y

X, y = Resample(X, y, scale=False, under=False, over=True)


"""Split events into training, validation and test sets."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#print(len(X_train), len(X_val), len(X_test)) 


"""Multiclassification to identify leptons. Very messy right now with many different algorithms"""
def getTrainScores(gs):
    # Function that plots the GridSearchCV best parameters and mean scores
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best


###Train models with various hyperparameters###

#param_grid = {"base_estimator__criterion": ["gini", "entropy"], "base_estimator__splitter": ["best", "random"], "n_estimators": [10, 100]}

model_DTC = DecisionTreeClassifier(random_state=42, max_features="auto", class_weight="balanced", max_depth=None)
model_ADC = AdaBoostClassifier(base_estimator=model_DTC, n_estimators=100, algorithm="SAMME.R", random_state=42)
model_Bag = BaggingClassifier(model_DTC, n_estimators=200, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)

#model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror","mlogloss"], num_class=4, n_jobs=-1, random_state=42, max_depth=8, reg_lambda=150, reg_alpha=20, n_estimators=150, importance_type="gain")  # Full Angular DF
model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=100, reg_alpha=10)  # 0.9638 0.9793 Full flat DF
#model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=300, reg_alpha=10)   # 0.9212 0.9408 Simple flat DF
model_XGBPipe = make_pipeline(StandardScaler(), model_XGB)

model_KNN = KNeighborsClassifier(n_neighbors=10, algorithm="ball_tree")
model_MLP = MLPClassifier(alpha=0.001, max_iter=200, random_state=42)
model_SVC = SVC(probability=True, gamma="auto", random_state=42)
model_SVC2 = SVC(decision_function_shape="ovr", random_state=42)
model_RF = Pipeline([("classifier", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, min_samples_split=8, max_depth=10, max_features="auto", random_state=42))])
model_RF2 = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, min_samples_split=5, max_depth=15, max_features="auto", random_state=42)
model_SGD = SGDClassifier()
model_GBC = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=None)
model_ONE_XGB = OneVsRestClassifier(XGBClassifier(objective="multi:softprob", eval_metric="merror", num_class=4, n_jobs=-1, random_state=42))
model_ONE_MLP = OneVsRestClassifier(model_MLP)

#grid_ADC = GridSearchCV(model_ADC, param_grid=param_grid, scoring="roc_auc")
#params={"reg_alpha":[1,20,80,120]}#,'reg_lambda':[1,50,100,150],"learning_rate":[1,0.1,0.01,5]}
#clf_XGB = GridSearchCV(estimator=model_XGB, param_grid=params, cv=3, n_jobs=-1, #scoring="f1_micro", verbose=10)
#clf_XGB.fit(X_train, y_train)
#getTrainScores(clf_XGB)


###Evaluate models with validation sets###
def eval_val(model, title):
    print("Start eval of models:")
    print(title)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    mae = mean_absolute_error(y_val, pred)
    #variance = np.mean(np.var(pred))
    #bias = np.mean((y_val - np.mean(pred))**2)
    #print("iter:",model.n_iter_)
    #print("outputs:",model.n_outputs_)
    #print("layers:",model.n_layers_)
    #print("classes:",model.classes_)
    #model_plt = plot_confusion_matrix(model, X_val, y_val, normalize="true")
    #model_plt.ax_.set_title(title)
    
    #skplt.estimators.plot_feature_importances(model, feature_names=features_names, title=title, x_tick_rotation=90)

    eval_val = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train], "MSE":[mse], "MAE":[mae]}#, "Var":[variance], "Bias":[bias]}
    return pd.DataFrame(eval_val)

#ADC_DF = eval_val(model_ADC, "AdaBoost")
#DTC_DF = eval_val(model_DTC, "DecisionTree")
#XGB_DF = eval_val(model_XGB, "XGBoost")
#Bag_XGB_DF = eval_val(model_Bag_XGB, "Bagging-XGB")
#Bag_DF = eval_val(model_Bag, "Bagging")
#KNN_DF = eval_val(model_KNN, "KNeighbors")
#MLP_DF = eval_val(model_MLP, "MLP")
#RF_DF = eval_val(model_RF, "RandomForestPipe")
##RF2_DF = eval_val(model_RF2, "RandomForest")
##SGD_DF = eval_val(model_SGD, "SGD")
#GBC_DF = eval_val(model_GBC, "GradientBoost")
#ONER_XGB_DF = eval_val(model_ONER_XGB, "OneVsRestXGB")
#ONE_MLP_DF = eval_val(model_ONE_MLP, "OneVsRestMLP")
#VC_DF = eval_val(model_VC, "Voting")
#df_merge = pd.concat([ADC_DF, XGB_DF])
#df_merge = pd.concat([ADC_DF, DTC_DF, XGB_DF, KNN_DF, MLP_DF, RF_DF])
#print(XGB_DF)
#print(df_merge)
#plt.show()



###Select best model and assess results with test set###

def eval_test(model, title):
    print("Assess final best " +title+ " model evaluation with test set:")
    if title == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_train = model.predict(X_train)
    """
    if title == "XGBoost":
        # Plot mlogloss and merror as function of iterations for train and test:
        print("Plot mlogloss and merror:")
        predictions = [round(value) for value in pred]
        new_acc = accuracy_score(y_test, predictions)
        results = model.evals_result()
        epochs = len(results["validation_0"]["merror"])
        x_axis = range(0,epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
        ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
        ax.legend()
        plt.title("mlogloss plot")
        plt.ylabel("log loss")
        fig,ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
        ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
        ax.legend()
        plt.title("merror plot")
        plt.ylabel("Error")
    """
    
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    variance = np.mean(np.var(pred))
    #bias = np.mean((y_test - np.mean(pred))**2)
    #precision = precision_score(y_test, pred, average="macro")
    prob = model.predict_proba(X_test)

    print(pred[0:10])
    print(prob[0:10])

    sys.exit()
    cks = cohen_kappa_score(y_test, pred)
    #cr = classification_report(y_test, pred)
    #cross_val = cross_val_score(model, X_test, y_test, cv=5)
    conf_mat = confusion_matrix(y_test, pred)

    #print("Precision:", precision)
    #print("Classification report "+title+": \n", cr)
    #print("Cross-validation score: \n", cross_val)
    #print("Highest accuracy score from cross-val:\n", np.max(cross_val))
    #print("Mean accuracy score from cross-val:\n", np.mean(cross_val))

    print(conf_mat)

    """
    lep1_pred_prob = prob[:,0]
    lep2_pred_prob = prob[:,1]
    lep3_pred_prob = prob[:,2]
    lep4_pred_prob = prob[:,3]

    plt.figure()
    _, bins, _ = plt.hist(lep1_pred_prob, bins=100, histtype="step", label="lep1", density=1)
    _, bins, _ = plt.hist(lep2_pred_prob, bins=bins, histtype="step", label="lep2", density=1)
    plt.legend(loc="best")
    """
    """
    print("Corr matrix:")
    plt.figure(figsize=(13,6))
    heatmap = sns.heatmap(features.corr(), annot=True, fmt=".2g", vmin=-1, vmax=1, center=0)
    heatmap.set_title("Correlation heatmap")
    """
    print("Conf matrix:")
    plot_confusion_matrix(model, X_test, y_test, normalize="true")
    """
    print("Importance as Series and sort:")
    plt.figure()
    importances = pd.Series(data=model.feature_importances_, index=features_names)#X_train.columns)
    importances_sorted = importances.sort_values()
    importances_sorted[-18:].plot(kind="barh")
    plt.tight_layout()
    
    print("Importance with skplt:")
    #plt.figure()
    skplt.estimators.plot_feature_importances(model, feature_names=features_names, title="XGB", x_tick_rotation=90, max_num_features=18)#feature=X_train.columns)
    plt.tight_layout()
    """
    """
    print("ROC:")
    #plt.figure()
    skplt.metrics.plot_roc(y_test, prob)

    print("Precision-Recall:")
    #plt.figure()
    skplt.metrics.plot_precision_recall(y_test, prob)
    
    if title == "XGBoost":
        print("XGB-importance:")
        #label_names = list(features.drop(columns=["target"], axis=1).columns)
        #model.get_booster().feature_names = features_names
        #xgb.plot_importance(model.get_booster(), importance_type="gain", title="Feature Importance - gain", show_values=False)
        xgb.plot_importance(model, importance_type="gain", title="Feature Importance - gain", show_values=False, max_num_features=18)
        plt.tight_layout()
    """
    """
    print("Elbow curve:")
    skplt.cluster.plot_elbow_curve(KMeans(random_state=42), X, cluster_ranges=range(2,20))
    
    print("Silhouette:")
    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit(X_train, y_train).predict(X_test)
    #cluster_labels = pred
    skplt.metrics.plot_silhouette(X_test, cluster_labels)

    silhouette_avg = silhouette_score(X_test, cluster_labels)
    sample_sil = silhouette_samples(X_test, cluster_labels)
    acc_kmean = accuracy_score(y_test, cluster_labels)
    print(acc_kmean)
    print(silhouette_avg)
    print(sample_sil)
    
    plt.figure()
    colors = cm.nipy_spectral(cluster_labels.astype(float)/4)
    plt.scatter(X_test[:,0], X_test[:,1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker="$%d$" %i, alpha=1, s=50, edgecolor="k")
    plt.title("Visual of clustered data")
    """
    plt.show()

    eval_test = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train], "CKS":[cks], "MSE":[mse], "MAE":[mae], "Var":[variance]}#, "Bias":[bias]}
    #print(eval_test)
    
    return pd.DataFrame(eval_test)

#print("Learning curve")
#skplt.estimators.plot_learning_curve(model_XGB, X, y, cv=5, shuffle=True, scoring="accuracy", n_jobs=-1, title="Learning curve")

#ADC = eval_test(model_ADC, "AdaBoost")
XGB_test_DF = eval_test(model_XGB, "XGBoost")
#print(pd.concat([ADC, XGB_test_DF]))
print(XGB_test_DF)
#plt.show()


