# CMPE 255 Final Project

## Group 4:
- Craig Belshe 017510974
- Krysle Beaulieu 007415772
- Nicolas Rios 017513691

## Assigned Data:
Census-Income Dta Set for '94, '95 - weighted census data from the US Census Bureau
https://archive.ics.uci.edu/dataset/117/census+income+kdd

## Goal:
The goal is to use the demographic data available in the data to determine whether the sample has 
an income of over $50,000 or under $50,000. This is interesting as it can lead to more information
on what attributes might predict income. 

It is a difficult task due to the data format and imbalance, with many categorical features and more than 90% of
samples under $50,000.


## Instructions to run:

All data is located in data/

1) cd Final_Classifier_Implementation
2) python3 histboost.py
  - There are parse args for modifying some parameters if desired. Options available with `-help`
  - Defaults run our optimal classifier
  - Prints out performance results with the classifiers predictions on group member data

e.g. `python3 histboost_final.py -r 42 -t .3 -l .1`

Default values, e.g. `python3 histboost_final.py`, will run with our winning parameters.


See readme in individual folders for how to run individual's code. 