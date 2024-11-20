import pandas as pd
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
# import xgboost as xgb
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# print(xgb.__version__)
  
# fetch dataset 
census_income_kdd = fetch_ucirepo(id=117) 
print('fetched the data')
  
# data (as pandas dataframes) 
X = census_income_kdd.data.features 
y = census_income_kdd.data.targets 

# choose what columns to use
# since we're predicting income greater/less than 50k, skipping columns which encode this info too directly:
# skipping, hourly wage, hours worked, capital gains, dividends, employed, 
columns_to_use = [
    'AAGE', # age, integer
    'ACLSWKR', # class of worker (private/government/etc), categorical
    'ADTINK', # industry code, integer
    'ADTOCC', # occupation code, integer
    'AHSCOL', # enrolled in that level of education this week, categorical
    'AHGA', # education level, integer, but actually categorical
    'AMJOCC', # Major occupation code, categorical
    'ARACE', # race, categorical
    'AREORGN', # hispanic, categorical (yes/no)
    'ASEX', # sex, categorical
    'AUNMEM', # member of labor union, categorical
    'PRCITSHP', # citizenship, integer, but actually categorical
    
    'CAPGAIN', # capital gains , integer
    'GAPLOSS', # capital losses, integer
    'DIVVAL', # dividends from stocks, integer
    'FILESTAT', # tax filer status, Categorical
    ]

columns_that_are_categorical = [ # definitely a better way to do this
    'ACLSWKR',
    'AHSCOL',
    'AHGA',
    'AMJOCC',
    'ARACE',
    'AREORGN',
    'ASEX',
    'AUNMEM',
    'PRCITSHP',

    'FILESTAT'
]

# X = X[columns_to_use]

X.drop(columns=[c for c in X.columns if c not in columns_to_use], inplace=True)
for col in X.columns:
    if col in columns_that_are_categorical:
        X[col] = X[col].astype('category') # set to the pandas category dype

    # can also encode all these features with
    # le = LabelEncoder()
    # X['education'] = le.fit_transform(X['education'])

# set y to be binary 0/1
y = y['income'].replace({'-50000': 0, ' 50000+.': 1}).astype(np.int64)

le = LabelEncoder()
for col in columns_that_are_categorical:
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print('class weight dict', class_weight_dict)

# manually weight class 1 more
# class_weight_dict[1] = class_weight_dict[1] + 37

# Create and train the Random Forest classifier
xgb_classifier = XGBClassifier(
    n_estimators=128,
    max_depth=12,
    learning_rate=0.1,
    # min_child_weight=2,
    # objective='multi:softmax',  # for multi-class classification
    # num_class=8,
    random_state=42
)

# adding in class weights from earlier
sample_weights = np.ones(y_train.shape[0], dtype='float')
for idx, label in enumerate(y_train):
    sample_weights[idx] = class_weight_dict[label]

# fit the model to the (train) data
xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)


from sklearn.metrics import accuracy_score, classification_report, f1_score

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy}", f"\nF1 Score: {f1}")

# Print detailed classification report
print(classification_report(y_test, y_pred))



"""
F1 Score: 0.6783676682361441
              precision    recall  f1-score   support

           0       0.98      0.89      0.93     37543
           1       0.29      0.76      0.42      2362

    accuracy                           0.88     39905
   macro avg       0.64      0.82      0.68     39905
weighted avg       0.94      0.88      0.90     39905

# adding in a few new features.
F1 Score: 0.7137353657146304
              precision    recall  f1-score   support

           0       0.99      0.90      0.94     37543
           1       0.34      0.82      0.48      2362

    accuracy                           0.90     39905
   macro avg       0.67      0.86      0.71     39905
weighted avg       0.95      0.90      0.92     39905

"""