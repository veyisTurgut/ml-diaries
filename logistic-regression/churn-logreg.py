import pandas
import seaborn
import matplotlib.pyplot as plt

##################### Importing all datasets
internet_data = pandas.read_csv("internet_data.csv")
customer_data = pandas.read_csv("customer_data.csv")
churn_data = pandas.read_csv("churn_data.csv")
###################### Merging all datasets based on condition ("customer_id ")
dataset = pandas.merge(pandas.merge(customer_data, internet_data, how="outer", on=["customerID"]), churn_data,
                       how="outer", on=["customerID"])
## Check for missing values
# print(dataset.isnull().sum())
## It appears that "TotalCharges" column is string type and has some empty entries. Lets handle this.
dataset["TotalCharges"].replace({" ": 0, "": 0}, inplace=True)
dataset["TotalCharges"] = pandas.to_numeric(dataset["TotalCharges"])

## Drop customer id
dataset = dataset.iloc[:, 1:]
## Explore data
# for column in dataset:
# print(dataset[column].value_counts())
# seaborn.countplot(x=column, hue="Churn", data=dataset)
# plt.show()
## As seen from the plot, gender is not making any difference, we can safely drop it.
dataset.drop(["gender"], axis=1, inplace=True)
## Get dummies
for column in ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
               "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]:
    newdf = pandas.get_dummies(dataset[column], drop_first=True, prefix=column)
    dataset.drop([column], axis=1, inplace=True)
    dataset = pandas.concat([dataset, newdf], axis=1)
del newdf
## Change "yes-no"s with 1-0
for column in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
    dataset[column].replace({"Yes": 1, "No": 0}, inplace=True)

###################### Scaling
from sklearn.preprocessing import StandardScaler

scaled_features = dataset.copy()
scale_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
features = scaled_features[scale_columns]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[scale_columns] = features
dataset = scaled_features
del scaled_features
###################### Train Test Split
y = dataset["Churn"]
x = dataset.drop(["Churn"], axis=1)
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
