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
    'PRCITSHP'
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

# note this is super imbalanced, should try SMOTE for that.


# split the data into train / val (there is a separte holdout set too)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('finished preprocessing and data split')
    
# Create and train the XGBoost model
# apparently XGBoost actually wants the data as integers even though they explicitly say they expect category type because reasons
# model = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42,
#     enable_categorical=True,  # Enable categorical feature support
#     tree_method='hist',
#     device='cuda'
# )

train_pool = Pool(X_train, y_train, cat_features=columns_that_are_categorical)
test_pool = Pool(X_test, y_test, cat_features=columns_that_are_categorical)

model = CatBoostClassifier(iterations=1000,
                           depth=6,
                           learning_rate=0.1,
                           loss_function='Logloss',
                           cat_features=columns_that_are_categorical,
                           task_type="GPU",
                           devices="0")

model.fit(train_pool)

# model.fit(X_train, y_train)
print('finished fitting the data')

# Make predictions
# y_pred = model.predict(test)
y_pred = model.predict(test_pool)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print out a conf matrix, and a good report of performance from sklearn that look good and include class by class performance and f1 score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature: {X.columns[i]}, Score: {v}")



# can also try CatBoost since it has a cool name and native support for categorical features.