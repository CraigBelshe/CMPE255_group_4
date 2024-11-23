import argparse
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder # preproc data
# allow load_data funcs
sys.path.append("../data/")
from load_data import load_ucimlrepo

# load data
print("loading data...")
X,y = load_ucimlrepo(True)
print("loading & encoded.")

# debug
print("feat keys: ",X.keys())
print("label keys: ",y.keys())
# end debug

#feature selection
from sklearn.feature_selection import chi2

# random forest imports
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from functools import partial # beacause I want to send in a param to f1 to cross val
import seaborn as sns

# search for the best
from sklearn.model_selection import GridSearchCV
params = {
    "n_estimators":[100,300,500,700],
    "max_features":["sqrt",None],
    "min_samples_leaf":[1,50,100,500],
    "criterion":['gini','entropy'],
}



def find_best_rf(params, X,y, n_jobs=-1, usef1=True):
    '''
    find the best random forest given a set of params.

    Parameters
    params: dict
        key, list(vals) of random forest key args to search for best score. 
    X: pandas df
        training data
    y: pandas df
        labels of training data
    n_jobs: int
        threads to use
    usef1: bool
        use f1 score if true. accuracy if false

    Returns:
    list of best features for the best random forest setup
    '''
    if usef1:
        oob = partial(f1_score,average='macro') # because accuracy by default for random forest and we don't want that

        rf = RandomForestClassifier(oob_score=oob,n_jobs=n_jobs,class_weight='balanced_subsample')
    else:
        rf = RandomForestClassifier(n_jobs=n_jobs,class_weight='balanced_subsample')

    rf_grid = GridSearchCV(estimator = rf, param_grid = params, cv = 7, verbose=4, n_jobs = 3,error_score=0)
    rf_grid.fit(X, np.ravel(y.to_numpy()))
    print("best params")
    print(rf_grid.best_params_.items())
    
    mdi_importances = pd.Series(rf_grid[-1].feature_importances_,).sort_values(ascending=True)
    print("imporantces of best rf: ",np.where(mdi_importances!=0))
    return rf_grid.best_params_.items()

def best_feats(X,y,n_jobs = -1,):
    ''' use a single random forest setup and get its best features.
    
    Parameters
    X: df
        training dataframe
    y: df
        training labels dataframe
    n_jobs: int
        number of threads
        
    returns:
        fitted random forest with f1 score'''
    oob = partial(f1_score,average='weighted') # because accuracy by default for random forest and we don't want that

    rf = RandomForestClassifier(oob_score=oob,n_jobs=n_jobs,class_weight='balanced_subsample',max_features="sqrt",min_samples_leaf=1,n_estimators=500,criterion="entropy")
    ravel_y  =np.ravel(y.to_numpy())
    rf.fit(X,ravel_y)
    mdi_importances = pd.Series(rf[-1].feature_importances_,).sort_values(ascending=False)
    col_names = []
    for i in np.where(mdi_importances!=0):
        col_names=[x for x in list(X.keys())[i]]
    print(mdi_importances)
    print("importances of best rf: ",col_names)
    return rf
    
def feat_selection(X,y,feat_func="chi2"):
    if feat_func=="chi2":
        _, p_vals = chi2(X.fillna(1000000),y) #toss in a useless value for nan
        # return the columns we are confident are highly important
        vals = np.where(p_vals<.05)
        print(f"using {len(vals[0])} features that we are confident are important via {feat_func}")
        return vals

vals = feat_selection(X,y)
ins = X[X.columns[vals]]
best_params = find_best_rf(params,ins,y)




# _ = best_feats(X[cols],y)
# SO many (macro/weighted f1, accuracy) runs have this a the best option: dict_items([('max_features', 'sqrt'), ('min_samples_leaf', 1), ('n_estimators', 500)])