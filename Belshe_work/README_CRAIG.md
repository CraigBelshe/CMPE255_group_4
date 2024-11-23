This folder contains some of the tests that were done with the following classifiers:
- Catboost
- Random Forest
- XGBoost
- Neural Network

The method of loading the data here is slightly different than in Krystle and Nic's folders. 
Here select features were selected from the dataset (rather than removing features from the dataset), and the resulting data is sent through a 
LabelEncoder instead of an OrdinalEncoder to convert categorical features to integers as applicable. Further, the test split here is 20/80 instead of 25/75.

A lot of these .py files are rather messy, but they should be mostly legible and work fine. catboost and xgboost dependencies may be needed.

The plot_chloropleth_usa_data.py was used to generate a chloropleth plot of the United States showing where each samples previous residence was.