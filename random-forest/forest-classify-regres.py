import pandas
import math

########################## CLASSIFICATION ########################
dataset = pandas.read_csv("bill_authentication.csv")
print(dataset.head())

### Preparing Data
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
### Training

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=5, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
### Evaluating

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = [round(x) for x in y_pred]
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
###########################################################################
###########################################################################
###########################################################################

########################## REGRESSION ########################

dataset = pandas.read_csv('petrol_consumption.csv')
print(dataset.head())

### Preparing Data
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

### Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
### Training

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
### Evaluating
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
