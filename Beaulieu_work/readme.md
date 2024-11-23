# How to run files

## random_forest.py
see `python3 random_forest.py --help` for more info but some examples to run are below:

1) `python3 random_forest.py -s True -n -1`

will use a smaller set of data to run faster (in <2 mins on an AMD 5950x). Uses all available cpu power for running this script.

chi2 to grab top features. 

grid search to find best rf.

will spit out top features from that then show some scores for the classes like f1 and accuracy.

or

2) `python3 random_forest.py` 

will use whole set of data to run more accurately (in 10s of minutes kind of range).

chi2 to grab top features. 

grid search to find best rf.

will spit out top features from that then show some scores for the classes like f1 and accuracy.
