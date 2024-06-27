#Importing the libraries
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import category_encoders as ce

data = pd.read_csv('bank.csv')

#Displaying the dataset
print("Dataset Shape: ", data.shape)
print("Dataset: ", data.head())

# Separating the target variable
X = data.drop(['y'], axis=1)
Y = data['y']

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
encoder = ce.OrdinalEncoder(cols=['job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','poutcome'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)

# Training Process
clf_gini.fit(X_train, y_train)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)

print("Predicted values:")
print(y_pred)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Accuracy : ",accuracy_score(y_test, y_pred)*100)
print("Report : ",classification_report(y_test, y_pred))
