

# Credit_Card_Fraud

Final project for the Building AI course

## Summary

The goal of this project is to predict if the user of the credit card is a fraud or not with a given dataset. 


## Background

This topic is important because we basicly pay for everything with credit cards so it important that your credit card is secure. This project aims to find if the credit card is used by someone who is not the owner.


## How is it used?

In order to use this project, a data should be given to the project so that it tells you if the user is a fraud or not.



Code Template:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import SCORERS, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
   dataset=pd.read_csv('Desktop/card_transdata.csv') #"Desktop/' part depends on the address that you have downloaded
   print(dataset)
   #Info
   print(dataset.head())
   print(dataset.info())
   print(dataset.describe())
   features=['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price','repeat_retailer','used_chip','used_pin_number','online_order']
   target=['fraud']
   X=np.array(dataset[features].values)
   y=np.array(dataset[target].values).ravel()
   #Normalization
   minX = X.min(axis=0)
   maxX = X.max(axis=0)
   X = (X - minX) / (maxX - minX)
   print("Normalization: ", X[:10])
   #Auto selecting
   feature_selection = SelectFromModel(LogisticRegression(tol=1e-1))
   feature_selection.fit(X, y)

   transformedX = feature_selection.transform(X)
   print(f"New shape: {transformedX.shape}")
   print("Selected features: ", feature_selection.get_support())
   print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

   feature_selection = SelectFromModel(LinearSVC(tol=1e-1))
   feature_selection.fit(X, y)

   transformedX = feature_selection.transform(X)
   print(f"New shape: {transformedX.shape}")
   print("Selected features: ", feature_selection.get_support())
   print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

   feature_selection = SelectFromModel(DecisionTreeClassifier())
   feature_selection.fit(X, y)

   transformedX = feature_selection.transform(X)
   print(f"New shape: {transformedX.shape}")

   print("Selected features: ", feature_selection.get_support())
   print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15) # 80% train, %20 test

   X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=15) # 50% val, 50% test
   print(f'Total # of sample in whole dataset: {len(X)}')
   print(f'Total # of sample in train dataset: {len(X_train)}')
   print(f'Total # of sample in validation dataset: {len(X_valid)}')
   print(f'Total # of sample in test dataset: {len(X_test)}')
   models = {
       'GaussianNB': GaussianNB(),
       'BernoulliNB': BernoulliNB(),
       'LogisticRegression': LogisticRegression(),
       'kNN': KNeighborsClassifier(n_neighbors=5),
   }

   for m in models:
     model = models[m]
     model.fit(X_train, y_train)
     score = model.score(X_valid, y_valid)
     print(f'{m} validation score => {score}')
   #kNN has the best validation score

   k_model=KNeighborsClassifier(n_neighbors=9)
   k_model.fit(X_train, y_train)

   validation_score = k_model.score(X_valid, y_valid)
   print(f'Validation score of trained model: {validation_score}')

   test_score = k_model.score(X_test, y_test)
   print(f'Test score of trained model: {test_score}')

   #Confusion Matrix
   y_predictions = k_model.predict(X_test)  #Predicting from the test data
   
   conf_matrix = confusion_matrix(y_test, y_predictions)
   print(f'Accuracy: {accuracy_score(y_test, y_predictions)}')
   print(f'Confussion matrix: \n{conf_matrix}\n')


main()
```


## Data source

[kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)



## Challenges

This project can only solve by using the given data and the features. And sometimes it may not predict correctly. You can see the accuracy if you run the code.

## What next?

To begin with, this project is not enough by itself. It can grow by adding some other algorithms and applying other methods such as deep learning (Convolutional neural networks).


