import numpy
import pandas
import matplotlib.pyplot as plt

############################ LINEAR REGRESSION BY HAND ############################
######### READ FILE
head_brain_dataset = pandas.read_csv("headbrain.csv")
X = head_brain_dataset["Head Size(cm^3)"].values
Y = head_brain_dataset["Brain Weight(grams)"].values
######### CALCULATE REGRESSION LINE
xbar = X.mean()
ybar = Y.mean()
print(xbar, ybar)
uppersum_of_m = 0
lowersum_of_m = 0
for i in range(len(X)):
    uppersum_of_m += (X[i] - xbar) * (Y[i] - ybar)
    lowersum_of_m += (X[i] - xbar) ** 2
m = uppersum_of_m / lowersum_of_m
print(m)
c = ybar - m * xbar
print(c)
######## PLOT LINE AND POINTS
max_x = X.max() + 100
min_x = X.min() - 100
x = numpy.linspace(min_x, max_x, 1000)
y = m * x + c
plt.plot(x, y, color="orange")
plt.scatter(X, Y)
plt.show()
######### CALCULATE R^2
uppersum_of_RSquare = 0
lowersum_of_RSquare = 0
for i in range(len(X)):
    #    uppersum_of_RSquare += (Y[i] - (m * X[i] + c)) ** 2
    uppersum_of_RSquare += (m * X[i] + c - ybar) ** 2
    lowersum_of_RSquare += (Y[i] - ybar) ** 2
# rsquare = 1 - uppersum_of_RSquare / lowersum_of_RSquare
rsquare = uppersum_of_RSquare / lowersum_of_RSquare
print(rsquare)

############################ LINEAR REGRESSION BY LIBRARIES ############################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# cannot use Rank 1 matrix in scikit learn
X = X.reshape((len(X), 1))
# create model
reg = LinearRegression()
# Fit training data
reg = reg.fit(X, Y)
Y_pred = reg.predict(X)
mse = mean_squared_error(Y, Y_pred)
rmse = numpy.sqrt(mse)
rsquare = reg.score(X, Y)
print(rmse)
print(rsquare)
