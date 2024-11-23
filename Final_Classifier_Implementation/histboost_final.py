"""
Group 4
Krystle Beaulieu
Craig Belshe
Nicolas Rios

This is an implementation of the Sklearn Histogram Gradient-Boosting Classifier to 
classify samples by income into <50,000 and >50,0000 using demographic census data.
It achieves a macro F1 score of 0.78, the best we saw.

This code may cause some Pandas warnings. These are due to changes in indexing Dataframes and can be ignored.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import sys
import argparse

# add the custom data loading script to path
sys.path.append('../data/')

from load_data import load_ucimlrepo

def parse_args():
    '''Arg parser for running this via command line. Defaults assume you run this file where it is living'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--random-state', type=int, default=17, help='Random state seed for more consistent results')
    parser.add_argument('-t', '--test-size', type=float, default=0.25, help='Percent of data to devote to the test set')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.14, help='Learning rate for training the classifier')

    return parser.parse_args()

def add_group_members_data(features):
    """"
    Add data from each team member, and two theoretical datas for older versions of two members. This is used to
    show how the classifier behaves on some real world data.
    """
    # As an interesting application of the classifier, it was run on data representing the group members.
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
    num_people = 6 # six items added
    return features, num_people

def preprocess_data(features, targets, random_state=17, num_people=6, test_size=0.25):
    """
    Preprocess the data by encoding categorical data and splitting according to the test split
    """
    print('Converting ordinal data to numerical...')
    for col in features:
        if features[col].nunique() < 53 or col == 'NOEMP':
            enc = OrdinalEncoder()
            features.loc[:,col] = enc.fit_transform(features[[col]])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features[:-num_people], targets, test_size=test_size, random_state=random_state)
    return Xtrain, Xtest, Ytrain, Ytest

def train_classifier(Xtrain, Ytrain, max_iter=200, random_state=17, learning_rate=0.14):
    """
    Creates a classifier using sklearn and trains using the training data.
    """

    model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=random_state, learning_rate=learning_rate)
    print('Fitting model...')
    model.fit(Xtrain, Ytrain)

    return model


def test_classifier(model, Xtest, Ytest, people_features):
    print('Predicting...')
    Ypred = model.predict(Xtest)
    Ypersonal = model.predict(people_features)

    f1 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='weighted')
    f2 = f1_score(Ytest.to_numpy().ravel(),Ypred,average='macro')
    acc = accuracy_score(Ytest.to_numpy().ravel(),Ypred)

    print('F1 score using weighted average is: {}'.format(f1))
    print('F1 score using macro average is: {}'.format(f2))
    print('Accuracy score is: {}'.format(acc))

    print(classification_report(Ytest, Ypred))
    print("Predictions on Team personal data:\n", Ypersonal)

def main(args):

    # load the dataset
    print('Loading data...')
    print('Replacing null values with column mode value...')
    features, targets = load_ucimlrepo(ordinal=False, dropnull=True)
    print('Data loaded.')
    
    # add data from group members
    features, num_people = add_group_members_data(features)

    # Preprocess and split data
    Xtrain, Xtest, Ytrain, Ytest = preprocess_data(
        features=features,
        targets=targets, 
        random_state=args.random_state, 
        num_people=num_people,
        test_size=args.test_size,
        )

    # Initialize and train the HistGradientBoosting classifier
    model = train_classifier(
        Xtrain,
        Ytrain,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
    )

    test_classifier(
        model,
        Xtest,
        Ytest,
        people_features=features[-num_people:]
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Done")
    



