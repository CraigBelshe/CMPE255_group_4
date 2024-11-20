import pandas as pd
import argparse

try:
    from ucimlrepo import fetch_ucirepo 
except ModuleNotFoundError:
    print("ucimlrepo not found. Please run pip install ucimlrepo and try this script again.")

def parse_args():
    '''arg parser for running this via command line. Fefaults assume you run this file where it is living'''
    parser = argparse.ArgumentParser()
    default_train_path = "data/census-income.data"
    default_labels_path = "data/census-income.names"
    default_test_path = "data/census-income.test"
    default_long_name = True

    parser.add_argument('--train_path', default=default_train_path, help='path to train data')
    parser.add_argument('--labels_path', default=default_labels_path, help='path to the .names labels file')
    parser.add_argument('--test_path', default=default_test_path, help='path to test data')
    parser.add_argument('--use_long_name',default=default_long_name, help='use the long name, eg. "class of worker". Else use the short name e.g. "ACLSWKR"')

    return parser.parse_args()
    
def load_ucimlrepo(ordinal=False):
    '''
    Load the data nicely via ucimlrepo. will only load the training data. 
    
    Parameters: None
    
    Returns: 
        census_income_kdd: dict
        dictionary of Census-Income Dta Set for '94, '95
        Can use census_income_kdd.data.features as training X and census_income_kdd.data.targets as training labels y. 
        This uses a class for income bucketed as > or < $50,000 income. Taken from "total person income"/"PTOTVAL"
    '''
    census_income_kdd = fetch_ucirepo(id=117) 
    df = census_income_kdd.data.features
    if ordinal:

        from sklearn.preprocessing import OrdinalEncoder
        for col in df:
            if df[col].nunique() < 53 or col == 'NOEMP':
                enc = OrdinalEncoder()
                df.loc[:,col] = enc.fit_transform(df[[col]])

    return df, census_income_kdd.data.targets
    
def load_feature_names(labels_path,headers_long=True):
    ''' 
    For manually loading the dataset. Loads feature names from census-income.names
    
    Returns: 
        pandas .dataframe of column names of the train/test set
    '''

    #c ode1-6 because it is seperated by \t but not the same number of \t of course
    df_names = pd.read_csv(labels_path, skiprows=23, nrows=44,header=None,sep="\t+",names=["feature",'shortname'])

    return df_names
    
def main(train_path, labels_path, test_path, headers_long=True):
    '''
    load the dataset and print some rows to show it loaded.

    Parameters:
    
    train_path: str
        Path to the census-income.data file
    labels_path: str
        Path to the census-income.names file
    test_path: str
        Path to the census-income.test file
    headers_long: bool
        If True, use the long header like "class of worker". 
        False will use the short name like "ACLSWKR". 
        Default is True.
    '''
    df_names = load_feature_names(labels_path,headers_long=True)
    # use the long name, eg. class of worker. Else use the short name e.g. ACLSWKR
    if headers_long:
        names = df_names['feature'].str.split("\| ", expand=True).loc[:,1]
    else: 
        names = df_names['shortname'].loc[:,1]
    
    df = pd.read_csv(train_path,delimiter=",", header=None, names=names)
    print(df.head())

    df_test = pd.read_csv(test_path,delimiter=",", header=None)
    print(df_test.head())

if __name__=="__main__":
    args = parse_args()
    print("option 1 to load: manually. Needs the short names fixed. See TODO in main.")
    main(args.train_path,args.labels_path,args.test_path,args.use_long_name)