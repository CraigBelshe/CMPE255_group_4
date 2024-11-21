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
print('Replacing null values with column mode value...')
features, targets = load_ucimlrepo(ordinal=False, dropnull=True)
print('Data loaded.')

print('Adding personal test feature sets...')

krystle_data = [32,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Married-civilian spouse present"," Other professional services"," Professional specialty","White","NA","Female","No","Not in universe","Full-time schedules",0,0,0,"Joint both under 65",' West',' California',' Householder',' Spouse of householder',0,'Nonmover','Nonmover','Nonmover',"Yes","No",0," Not in universe","United-States","United-States","United-States","Native- Born in the United States",0,"No",0,52,0,94]
nic_data = [25,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Never married"," Other professional services"," Professional specialty","White","Central or South American","Male","No","Not in universe","Full-time schedules",0,0,0,"Single",' South',' Georgia',' Householder',' Nonrelative of householder',1830.11,'Nonmover','Nonmover','Nonmover',"Yes"," Not in universe",6," Columbia","Columbia","United-States","United-States","Native- Born in the United States",0," Not in universe",0,52,0,94]
krystle_man_data = [32,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Married-civilian spouse present"," Other professional services"," Professional specialty","White","NA","Male","No","Not in universe","Full-time schedules",0,0,0,"Joint both under 65",' West',' California',' Householder',' Spouse of householder',0,'Nonmover','Nonmover','Nonmover',"Yes","No",0," Not in universe","United-States","United-States","United-States","Native- Born in the United States",0,"No",0,52,0,94]
krystle_man_old_data = [50,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Married-civilian spouse present"," Other professional services"," Professional specialty","White","NA","Male","No","Not in universe","Full-time schedules",0,0,0,"Joint both under 65",' West',' California',' Householder',' Spouse of householder',0,'Nonmover','Nonmover','Nonmover',"Yes","No",0," Not in universe","United-States","United-States","United-States","Native- Born in the United States",0,"No",0,52,0,94]
krystle_woman_old_data = [50,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Married-civilian spouse present"," Other professional services"," Professional specialty","White","NA","Female","No","Not in universe","Full-time schedules",0,0,0,"Joint both under 65",' West',' California',' Householder',' Spouse of householder',0,'Nonmover','Nonmover','Nonmover',"Yes","No",0," Not in universe","United-States","United-States","United-States","Native- Born in the United States",0,"No",0,52,0,94]
craig_data = craig_data = [25,"Private",36,4,"Bachelors degree(BA AB BS)","College or university","Never married"," Other professional services"," Professional specialty","White","NA","Male","No","Not in universe","Full-time schedules",0,0,1500,"Single",' West',' California',' In group quarters',' Group Quarters- Secondary individual',1830.11,'Nonmover','Nonmover','Nonmover',"No"," Not in universe",0," Not in universe","United-States","United-States","United-States","Native- Born in the United States",0," Not in universe",0,52,0,94]

features.loc[len(features)]=krystle_data
features.loc[len(features)]=nic_data
features.loc[len(features)]=krystle_man_data
features.loc[len(features)]=krystle_man_old_data
features.loc[len(features)]=krystle_woman_old_data
features.loc[len(features)]=craig_data

print('Converting ordinal data to numerical...')
for col in features:
    if features[col].nunique() < 53 or col == 'NOEMP':
        enc = OrdinalEncoder()
        features.loc[:,col] = enc.fit_transform(features[[col]])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features[:-6], targets, test_size=0.25, random_state=17)

model = HistGradientBoostingClassifier(max_iter=200, random_state=17, learning_rate=0.14)
print('Fitting model...')
model.fit(Xtrain, Ytrain)
print('Predicting...')
Ypred = model.predict(Xtest)
Ypersonal = model.predict(features[-6:])

f1 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='weighted')
f2 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='macro')
acc = accuracy_score(Ytest.to_numpy().ravel(),Ypred)

_, predcounts = np.unique(Ypred, return_counts=True)
_, counts = np.unique(targets, return_counts=True)

predpercent = predcounts[0]/(predcounts[0] + predcounts[1])
truepercent = counts[0]/(counts[0] + counts[1])

print('F1 score using weighted average is: {}'.format(f1))
print('F1 score using macro average is: {}'.format(f2))
print('Accuracy score is: {}'.format(acc))
print('Percentage of test set predicted to be > 50000: {}'.format(predpercent))
print('Percentage of total set actually > 50000: {}'.format(truepercent))
print(Ypersonal)