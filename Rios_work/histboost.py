import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../data/')

from load_data import load_ucimlrepo

print('Loading data...')
print('Converting ordinal data to numerical...')
print('Replacing null values with column mode value...')
features, targets = load_ucimlrepo(ordinal=True, dropnull=True)
print('Data loaded.')


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, targets, test_size=0.3, random_state=17)

model = HistGradientBoostingClassifier(max_iter=200, random_state=17)
print('Fitting model...')
model.fit(Xtrain, Ytrain)
print('Predicting...')
Ypred = model.predict(Xtest)

f1 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='weighted')
f2 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='macro')
acc = accuracy_score(Ytest.to_numpy().ravel(),Ypred)

print('F1 score using weighted average is: {}'.format(f1))
print('F1 score using macro average is: {}'.format(f2))
print('Accuracy score is: {}'.format(acc))