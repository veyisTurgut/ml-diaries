import pandas

suv_dataset = pandas.read_csv("suv_data.csv")
sex = pandas.get_dummies(data=suv_dataset["Gender"], drop_first=True)
suv_dataset = pandas.concat([suv_dataset, sex], axis=1)
suv_dataset.drop(["Gender"], axis=1, inplace=True)
## Take age, salary and gender as params
x = suv_dataset.iloc[:, [1, 2, 4]].values
y = suv_dataset.iloc[:, 3].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
## Scale the values for better performance
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred) * 100)
