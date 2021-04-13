import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import scikitplot as skplt
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Tools
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.evaluate import bias_variance_decomp
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.experimental import enable_hist_gradient_boosting  # Needed when importing the HistGradientBoostingClassifier, experimental feature
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, precision_score, cohen_kappa_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Models
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  # Very slow to run
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")#, category=UserWarning)


"""Read dataframe(s) from file."""
df_flat = pd.read_hdf("Trilepton_ML.h5", key="DF_flat3")
df_flat = df_flat.select_dtypes(exclude=["int32"])

# Make a simplified dataframe with only pt, phi and eta for each lepton
df_simple = df_flat.select_dtypes(exclude=["int32", "float64"])
df_simple = df_simple.drop(["lep1_theta", "lep1_px", "lep1_py", "lep1_pz", "lep1_E", "lep2_theta", "lep2_px", "lep2_py", "lep2_pz", "lep2_E", "lep3_theta", "lep3_px", "lep3_py", "lep3_pz", "lep3_E", "lep4_theta", "lep4_px", "lep4_py", "lep4_pz", "lep4_E"], axis=1)


"""Make design matrix X and target y from chosen dataframe."""
features = df_simple
#features = df_flat

X = features.select_dtypes(exclude=["object"])
features_names = list(X.columns)
Y = features.target

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

# Inspect the data
def data_info(df):
    df.info()
    print(df.head())
    print(y.target.value_counts())  # Print the counts of the different classes
    
    # Look at the correlations between the features
    print("Corr matrix:")
    plt.figure(figsize=(13,6))
    heatmap = sns.heatmap(df.corr(), annot=True, fmt=".2g", vmin=-1, vmax=1, center=0)
    heatmap.set_title("Correlation heatmap")
    
    # Information gain of the features
    cols = list(X.columns)
    infos = mutual_info_classif(X.values, np.ravel(y), random_state=42)
    info_gain = {}
    for i in range(len(cols)):
        info_gain[str(cols[i])] = infos[i]
    
    info_gain = pd.DataFrame(info_gain, index=[0])
    print("Information gain of the features:")
    print(info_gain)
    
    plt.show()
    
#data_info(features)


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
    
    #print(y.target.value_counts())  # Print the counts of the different classes after resampling
    return X, y 

X, y = Resample(X, y, under=True, over=True)

"""
kbest=15
selectK = SelectKBest(f_classif,k=kbest)
selectK.fit(X,y)
X_sel = selectK.transform(X)
#features = X.columns[selectK.get_support()]
print('Select {} best features using f_classif'.format(kbest))
"""


"""Split events into training, validation and test sets."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


"""Scale the data when called."""
def scaler(X_train, X_val, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    return X_train, X_val, X_test

X_train, X_val, X_test = scaler(X_train, X_val, X_test)

#print(len(X_train), len(X_val), len(X_test)) 


"""Multiclass classification to identify leptons."""
def getTrainScores(gs):
    # Function that prints the RandomizedSearchCV best parameters and mean scores
    gs.fit(X_train, y_train)
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best


#Train models with various hyperparameters for optimization
#param_grid = {"base_estimator__criterion": ["gini", "entropy"], "base_estimator__splitter": ["best", "random"], "n_estimators": [10, 100]}

model_LogReg = LogisticRegression(multi_class="multinomial", random_state=42, n_jobs=-1, max_iter=1000)
model_LogRegCV = LogisticRegression(multi_class="multinomial", random_state=42, n_jobs=-1, max_iter=1000)
model_Ridge = RidgeClassifier(random_state=42)
model_RidgeCV = RidgeClassifierCV()

model_DTC = DecisionTreeClassifier(random_state=42, max_features="auto", class_weight="balanced", max_depth=18)
model_ADC = AdaBoostClassifier(base_estimator=model_DTC, n_estimators=250, algorithm="SAMME.R", random_state=42)
model_Bag = BaggingClassifier(model_DTC, n_estimators=300, max_samples=500, bootstrap=True, n_jobs=-1, random_state=42)

#model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=100, reg_alpha=10, early_stopping_rounds=50)  # 0.9638 0.9793 Full flat DF

model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=500, reg_alpha=10, learning_rate=0.2, subsample=0.9, colsample_bytree=0.85, min_child_weight=3, max_depth=10)  # Test pruning Full flat DF 


#model_XGB = XGBClassifier(objective="multi:softprob", eval_metric=["merror", "mlogloss"], num_class=6, n_jobs=-1, importance_type="gain", random_state=42, n_estimators=300, reg_alpha=10)   # 0.9212 0.9408 Simple flat DF
model_XGBPipe = make_pipeline(PCA(), model_XGB)

model_LGBM = LGBMClassifier(objective="multiclass", num_class=6, n_estimators=500, max_depth=8, learning_rate=0.1, random_state=42, metric=["multi_logloss", "multi_error"])

model_MLP = MLPClassifier(alpha=0.1, max_iter=300, activation="relu", hidden_layer_sizes=(200,200,100,100,50), random_state=42) # better with scaling
model_SVC = SVC(probability=True, gamma="auto", random_state=42)
model_SVC2 = SVC(decision_function_shape="ovr", random_state=42)
model_RF_Pipe = Pipeline([("classifier", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, min_samples_split=5, max_depth=20, max_features="auto", random_state=42))])
model_RF = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, min_samples_split=5, max_depth=15, max_features="auto", random_state=42)
model_GBC = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5, loss="deviance")
model_HistGBC = HistGradientBoostingClassifier(random_state=42, max_iter=200, max_depth=5, loss="categorical_crossentropy")
model_ONE_XGB = OneVsRestClassifier(XGBClassifier(objective="multi:softprob", eval_metric="merror", num_class=6, n_jobs=-1, random_state=42))
model_ONE_MLP = OneVsRestClassifier(model_MLP)
model_VC = VotingClassifier(estimators=[("DTC", model_DTC), ("Ada", model_ADC),("RF", model_RF), ("HistGBC", model_HistGBC), ("XGB", model_XGB)], voting="soft", n_jobs=-1)

#grid_ADC = RandomizedSearchCV(model_ADC, param_grid=param_grid, scoring="roc_auc")
#params={"reg_alpha":[5,10,20,60]}#,'reg_lambda':[1,50,100,150],"learning_rate":[1,0.1,0.01,5]}
#clf_XGB = RandomizedSearchCV(estimator=model_XGB, param_distributions=params, cv=3, n_jobs=-1, scoring="accuracy", random_state=42)
#getTrainScores(clf_XGB)





#Evaluate models with validation set
def eval_val(model, title):
    print("Start eval of model "+title+":")
    if title == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train, y_train)

    pred = model.predict(X_val)
    pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    mae = mean_absolute_error(y_val, pred)
    variance = np.mean(np.var(pred))
    bias = np.mean((y_val.values - np.mean(pred))**2)
    #print("iter:",model.n_iter_)
    #print("outputs:",model.n_outputs_)
    #print("layers:",model.n_layers_)
    #print("classes:",model.classes_)
    if title == "XGBoost" or title == "LGBM":
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

    model_plt = plot_confusion_matrix(model, X_val, y_val, normalize="true")
    model_plt.ax_.set_title(title)

    #skplt.estimators.plot_feature_importances(model, feature_names=features_names, title=title, x_tick_rotation=90)

    eval_val = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train], "MSE":[mse], "MAE":[mae], "Var":[variance], "Bias":[bias]}
    return pd.DataFrame(eval_val)
"""
DTC_DF = eval_val(model_DTC, "DecisionTree")
ADC_DF = eval_val(model_ADC, "AdaBoost")
#XGB_DF = eval_val(model_XGB, "XGBoost")
LGBM_DF = eval_val(model_LGBM, "LGBM")
#Bag_XGB_DF = eval_val(model_Bag_XGB, "Bagging-XGB")
#Bag_DF = eval_val(model_Bag, "Bagging")
MLP_DF = eval_val(model_MLP, "MLP")
#RF_DF = eval_val(model_RF, "RandomForest")
#RF2_DF = eval_val(model_RF_Pipe, "RandomForestPipe")
#GBC_DF = eval_val(model_GBC, "GradientBoost")
#ONER_XGB_DF = eval_val(model_ONER_XGB, "OneVsRestXGB")
#ONE_MLP_DF = eval_val(model_ONE_MLP, "OneVsRestMLP")
#VC_DF = eval_val(model_VC, "Voting")
#df_merge = pd.concat([DTC_DF, MLP_DF])
#LogReg_DF = eval_val(model_LogReg, "LogReg")
#LogRegCV_DF = eval_val(model_LogRegCV, "LogRegCV")
#Ridge_DF = eval_val(model_Ridge, "Ridge")
#Ridge_DF = eval_val(model_RidgeCV, "RidgeCV")
df_merge = pd.concat([DTC_DF, MLP_DF, ADC_DF])
#print(XGB_DF)
print(df_merge)
#plt.show()
"""


#Select best model and assess results with test set
def eval_test(model, title):
    print("Assess final best " +title+ " model evaluation with test set:")
    if title == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_train = model.predict(X_train)

    accuracy_train = accuracy_score(y_train, pred_train)
    accuracy = accuracy_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    variance = np.mean(np.var(pred))
    bias = np.mean((y_test.values - np.mean(pred))**2)
    precision = precision_score(y_test, pred, average="macro")
    prob = model.predict_proba(X_test)
    prob_train = model.predict_proba(X_train)

    
    print("Kolmogorov-Smirnoff statistics:")
    i = 0
    KS_stat = ks_2samp(prob[:,i], prob_train[:,i])
    print(KS_stat)
    def empirical_cdf(sample, plotting=True):
        N = len(sample)
        rng = max(sample) - min(sample)
        if plotting:
            xs = np.concatenate([np.array([min(sample)-rng/3]), np.sort(sample) , np.array([max(sample)+rng/3])])
            ys = np.append(np.arange(N+1)/N, 1)
        else:
            xs = np.sort(sample)
            ys = np.arange(1, N+1)/N
        return (xs, ys)
    xs_test, ys_test = empirical_cdf(prob[:,i])
    xs_train, ys_train = empirical_cdf(prob_train[:,i])
    plt.figure()
    plt.plot(xs_test, ys_test, label="Test", linewidth=3,linestyle=":")
    plt.plot(xs_train, ys_train, label="Train")
    plt.title("Kolmogorov-Smirnoff statistics")
    plt.xlabel("Score")
    plt.ylabel("c.d.f.")
    plt.legend()
    
    
    cks = cohen_kappa_score(y_test, pred)
    cr = classification_report(y_test, pred)
    #cross_val = cross_val_score(model, X_test, y_test, cv=5)
    conf_mat = confusion_matrix(y_test, pred)

    #print("Precision:", precision)
    #print("Classification report "+title+": \n", cr)
    #print("Cross-validation score: \n", cross_val)
    #print("Highest accuracy score from cross-val:\n", np.max(cross_val))
    #print("Mean accuracy score from cross-val:\n", np.mean(cross_val))

    """
    #Unsure?
    y1_pred_prob = prob[:,0]
    y2_pred_prob = prob[:,1]
    y3_pred_prob = prob[:,2]
    y4_pred_prob = prob[:,3]
    y5_pred_prob = prob[:,4]
    y6_pred_prob = prob[:,5]
    
    plt.figure()
    plt.hist(y1_pred_prob, bins=10)
    plt.figure()
    plt.hist(y2_pred_prob, bins=10)
    plt.figure()
    plt.hist(y3_pred_prob, bins=10)
    plt.figure()
    plt.hist(y4_pred_prob, bins=10)
    plt.figure()
    plt.hist(y5_pred_prob, bins=10)
    plt.figure()
    plt.hist(y6_pred_prob, bins=10)
    """
    """
    class_names = ["123", "132", "213", "231", "312", "321"]
    for i in range(len(class_names)):
        plt.figure()
        plt.hist(model.predict_proba(X_test)[:,i], bins=100, label="Test %s" %class_names[i], alpha=0.5, density=1)
        plt.hist(model.predict_proba(X_train)[:,i], bins=100, label="Train %s" %class_names[i], alpha=0.5, density=1, histtype="step")
    #_, bins, _ = plt.hist(y2_pred_prob, bins=bins, histtype="step", label="lep2", density=1)
    plt.legend(loc="best")
    plt.show()
    """
    #sys.exit()
    
    #print("Conf matrix:")
    #print(conf_mat)
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

        print("XGB-tree:")
        xgb.plot_tree(model, rankdir="LR")
    """
    """
    Unsure?
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
    #plt.show()

    eval_test = {"Model":[title], "Score":[accuracy], "Score_train":[accuracy_train],"CKS":[cks], "MSE":[mse], "MAE":[mae], "Var":[variance], "Bias":[bias]}
    #print(eval_test)
    
    return pd.DataFrame(eval_test)

"""
mse, bias, var = bias_variance_decomp(model_DTC, X_train, y_train.values, X_test, y_test.values, loss='mse', num_rounds=10, random_seed=42)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
"""

#XGB_test_DF = eval_test(model_XGB, "XGBoost")
HistGBC_DF = eval_test(model_HistGBC, "HGBC")
LGBM_DF = eval_test(model_LGBM, "LGBM")
#VC_DF = eval_test(model_VC, "VC")
#XGB_Pipe_DF = eval_test(model_XGBPipe, "XGBoostPipe")
merge = pd.concat([HistGBC_DF, LGBM_DF])
#print(XGB_test_DF)
print(merge)

