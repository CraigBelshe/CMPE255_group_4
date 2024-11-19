import argparse
import pandas as pd
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
print(X.keys())
print(y.keys())
# end debug



# random forest imports
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score ,RepeatedStratifiedKFold,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

# metrics
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from functools import partial # beacause I want to send in a param to f1 to cross val
import seaborn as sns

# search for the best
from sklearn.model_selection import RandomizedSearchCV
#     "criterion":['gini','entropy'],
#     "min_samples_leaf":[1,2,3,4],
params = {
    "n_estimators":[50,100,500,1000],
    "max_features":["log2","sqrt",None],
}
oob = partial(f1_score,average='weighted') # because accuracy by default for random forest
n_jobs = -1


rf = RandomForestClassifier(oob_score=oob,n_jobs=n_jobs,class_weight='balanced_subsample')
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = params, n_iter = 20, cv = 3, verbose=4, random_state=42, n_jobs = -1,error_score=0)
rf_random.fit(X, y)