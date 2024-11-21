# How to run files

## random_forest.py
`python3 random_forest.py --find_best`

will find the best random forest out of a set of parameters via grid search
(parameters are hard coded, see parms inside .py file for details)

or

`python3 random_forest.py` 

will train on the training dataset with bootstrap and one of the common best params (500 est, entropy, 1 min leaf, sqrt max feat)