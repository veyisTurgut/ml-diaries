import pandas
import seaborn
import matplotlib.pyplot as plt

dataset = pandas.read_csv("bank-additional-full.csv")
print(dataset.shape)
# print(dataset.head())
## Check for missing values
# print(dataset.isnull().sum())
## Wow! Such a nice dataset

## Explore data
# for column in dataset:
#     print(dataset[column].value_counts())
#     seaborn.countplot(x=column, hue="y", data=dataset)
#     plt.show()

## Some observations:
# day_of_week and paydays has no effect at all
# We can drop them safely.
# There are some "unknown"s , we can delete those lines too.
##########
dataset.drop(["day_of_week", "pdays"], axis=1, inplace=True)
dataset.drop(dataset[
                 (dataset["marital"] == "unknown") | (dataset["job"] == "unknown") | (
                         dataset["education"] == "unknown") | (dataset["default"] == "unknown") | (
                         dataset["housing"] == "unknown") |
                 (dataset["loan"] == "unknown")].index,
             inplace=True)
# print(dataset.shape)
## Explore data
# for column in dataset:
#     print(dataset[column].value_counts())
#     seaborn.countplot(x=column, hue="y", data=dataset)
#     plt.show()
###################### Scaling
### We should scale float and integer values

from sklearn.preprocessing import StandardScaler

scaled_features = dataset.copy()
scale_columns = ["age", "duration", "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
                 "nr.employed"]
features = scaled_features[scale_columns]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[scale_columns] = features
dataset = scaled_features
del scaled_features

## Replace "yes-no"s with 0-1 and get dummies for ( encode ) string values
dataset.replace({"yes": 1, "no": 0}, inplace=True)
for column in ["job", "marital", "education", "contact", "month", "poutcome"]:
    newdf = pandas.get_dummies(dataset[column], drop_first=True, prefix=column)
    dataset.drop([column], axis=1, inplace=True)
    dataset = pandas.concat([dataset, newdf], axis=1)
del newdf
# print(dataset.value_counts())

###################### Train Test Split
y = dataset["y"]
x = dataset.drop(["y"], axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

###################### Model Building
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
###################### Getting the predicted values on train set
predictions = logreg.predict(x_test)
###################### Create a confusion matrix on train set and test
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
###################### Check the overall accuracy
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
