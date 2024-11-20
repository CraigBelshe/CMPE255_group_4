import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
# import xgboost as xgb
# from catboost import CatBoostClassifier
# from catboost import Pool
from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
import keras
# from keras.utils import FeatureSpace, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Calculate class weights
# class_weights = {0: 1, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
# class_weights={0: 0.17, 1: 0.23, 2: 1.01, 3: 6.01, 4: 8.92, 5: 98.08, 6: 33.95, 7: 2.08, 8: 0, 9: 0, 10: 0}

# Step 3: Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("Class Weights:", class_weight_dict)

print('# of features', X_train.shape[1])
# Create the model
# model = Sequential([
#     # Dense(64, activation='relu', input_shape=(train_dataframe.shape[1],)),
#     # Dense(32, activation='relu'),
#     # Dense(8, activation='softmax')
#     Dense(128, activation='relu', input_shape=(train_dataframe.shape[1],)),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     # Dense(64, activation='relu'),
#     # Dropout(0.3),
#     Dense(8, activation='softmax')
# ])
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])

opt = keras.optimizers.Adam(learning_rate=0.001, )
from tensorflow.keras.metrics import F1Score
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1Score(threshold=0.5,average='macro')])


# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    # validation_split=0.2,
    validation_data=(X_test_scaled, y_test),
    class_weight=class_weight_dict
)

y_pred = np.argmax(model.predict(X_test_scaled), axis=-1)  # Get predicted classes

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))