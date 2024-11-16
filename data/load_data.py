import pandas as pd
import argparse

def parse_args():
    '''arg parser for running this via command line. Fefaults assume you run this file where it is living'''
    parser = argparse.ArgumentParser()
    default_train_path = "./census-income.data"
    default_labels_path = "./census-income.names"
    default_test_path = "./census-income.test"
    default_long_name = True

    parser.add_argument('--train_path', default=default_train_path, help='path to train data')
    parser.add_argument('--labels_path', default=default_labels_path, help='path to the .names labels file')
    parser.add_argument('--test_path', default=default_test_path, help='path to test data')
    parser.add_argument('--use_long_name',default=default_long_name, help='use the long name, eg. "class of worker". Else use the short name e.g. "ACLSWKR"')

    return parser.parse_args()

def main(train_path, labels_path, test_path,headers_long=True):
    df_names = pd.read_csv('../data/census-income.names', skiprows=23, nrows=44,header=None,sep="\t",names=["feature",'code1','code2','code3','code4','code5','code6'])

    # use the long name, eg. class of worker. Else use the short name e.g. ACLSWKR
    if headers_long:
        names = df_names['feature'].str.split("\| ", expand=True).loc[:,1]
    else:
        names = "TODO" # TODO fix the code1-6 bits to collapse it and make it the short names. 
    
    df = pd.read_csv("../data/census-income.data",delimiter=",",header=None,names=names)
    print(df.head())
    print("TODO, MORE THINGS")

if __name__=="__main__":
    args = parse_args()
    main(args.train_path,args.labels_path,args.test_path,args.use_long_name)