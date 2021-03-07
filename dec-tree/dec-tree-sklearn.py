import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################## CLASSIFICATION ########################
dataset = pd.read_csv("bill_authentication.csv")
### Data Analysis

print(dataset.shape)
print(dataset.head())

### Prepare Data

x = dataset.drop('Class', axis=1)
y = dataset['Class']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

### Training and Making Predictions

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

## Evaluate

from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

print()
print()
print()
print()
print()

######################## REGRESSION ########################
dataset = pd.read_csv('petrol_consumption.csv')

### Data Analysis

print(dataset.shape)
print(dataset.head())

### Prepare Data

X = dataset.drop('Petrol_Consumption', axis=1)
y = dataset['Petrol_Consumption']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Training and Making Predictions
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

### Comparison
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

## Evaluate

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
