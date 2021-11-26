
import pandas as pd

from sklearn.datasets import load_boston
boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)
# Starting Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
print("Intercept:" + str(model.intercept_.round(2)))
print("Slope:" + str(model.coef_.round(2)))

# Setting a predicting Y variable
y_test_predicted = model.predict(X_test)
#Show the prediction # Write in np.array() a number you want to predict
import numpy as np
new_y = np.array([6.5]).reshape(-1,1)
print("Predict:" + str(model.predict(new_y)))

#Counting residuals
residuals = Y_test - y_test_predicted

#Import overview library
from sklearn.metrics import mean_squared_error
#R2 Score
print("R2score: " + str((model.score(X_test, Y_test).round(3))*100) + " %")
#MSE
from sklearn.metrics import mean_squared_error
print("MSE:" + str(mean_squared_error(Y_test, y_test_predicted).round(2)))


