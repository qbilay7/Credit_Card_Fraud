#Goal is to predict if the user of the credit card is a fraud or not
#It can be used in bank companies
#Dataset is from kaggle.com
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
    'SVC': SVC(kernel='rbf'),
    'kNN': KNeighborsClassifier(n_neighbors=5),
}

for m in models:
  model = models[m]
  model.fit(X_train, y_train)
  sscore = model.score(X_valid, y_valid)
  print(f'{m} validation score => {sscore}')
#GaussianNB has the best validation score

k_model=GaussianNB()
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

