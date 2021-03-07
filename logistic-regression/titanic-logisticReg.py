import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

################## Import Data ##################
titanic_dataset = pandas.read_csv("titanic-trainingdataset.csv")

################## Some Analyze ##################
# seaborn.countplot(x="Survived", data=titanic_dataset)
# plt.show()
# seaborn.countplot(x="Survived", hue="Sex", data=titanic_dataset)
# plt.show()
# seaborn.countplot(x="Survived", hue="Pclass", data=titanic_dataset)
# plt.show()
# titanic_dataset["Age"].plot.hist()
# plt.show()
# titanic_dataset["Fare"].plot.hist()
# plt.show()
# print(titanic_dataset.info())
# seaborn.countplot(x="SibSp", data=titanic_dataset)
# plt.show()

################## Clean Data ##################
# print(titanic_dataset.isnull())
##check null values
# print(titanic_dataset.isnull().sum())
## heatmap of null values
seaborn.heatmap(titanic_dataset.isnull(), yticklabels=False, cmap="viridis")
# plt.show()
seaborn.boxplot(x="Pclass", y="Age", data=titanic_dataset)
# plt.show()
## Drop "Cabin" column, cause it has a lot of null values
titanic_dataset.drop("Cabin", axis=1, inplace=True)
## Drop all null values
titanic_dataset.dropna(inplace=True)
seaborn.heatmap(titanic_dataset.isnull(), yticklabels=False, cmap="viridis")
# plt.show()
# print(titanic_dataset.isnull().sum())
## We want to work with binaries, so represent string columns with numbers. To give a concrete example, convert "male-female" column to "0-1" column where 1 represents male.
sex = pandas.get_dummies(titanic_dataset["Sex"], drop_first=True)
## At first embarked has 3 values: "S-Q-C", we convert this to two columns. Q and S. When both of them 0, C becomes 1.
embark = pandas.get_dummies(titanic_dataset["Embarked"], drop_first=True)
## At first Pclass has 3 values: "1-2-3", we convert this to two columns. 2 and 3. When both of them 0 means 1.
pclass = pandas.get_dummies(titanic_dataset["Pclass"], drop_first=True)
## concatenate new columns
titanic_dataset = pandas.concat([titanic_dataset, sex, embark, pclass], axis=1)
## drop old-unnecessary columns
titanic_dataset.drop(["Sex", "Embarked", "Pclass", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)
# print(titanic_dataset.info())

################## Train Data ##################
y = titanic_dataset["Survived"]
x = titanic_dataset.drop("Survived", axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
predictions = logreg.predict(x_test)
################## Accuracy Check ##################
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
