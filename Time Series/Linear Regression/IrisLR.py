import pandas as pd
from sklearn import datasets
import numpy as np
irisdata = datasets.load_iris()
df = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
df["species"] = irisdata.target_names[irisdata.target]

X = df[['petal length (cm)']]
Y = df['sepal length (cm)']

print(df)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print(X_train.shape)
print(Y_train.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

input = np.array([16]).reshape(-1,1)
print("Prediction: " + str(model.predict(input)))
y_predicted = model.predict(X_test)
residuals = y_predicted - Y_test
model.intercept_.round(2)


#Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

names = {'setosa': 0,
'versicolor': 1,
'virginica': 2}
colors = [names[item] for item in df['species']]


plt.scatter(X, Y, c = colors)
plt.plot(X_train, model.predict(X_train), color='lightcoral')
plt.show()

#Model Overview
print("R2score: " + str((model.score(X_test, Y_test).round(3))*100) + " %")

#OSL Method
import statsmodels.formula.api as smf
import statsmodels.api as sm
X = sm.add_constant(X)


results = sm.OLS(Y, X).fit()
print(results.summary())


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(results, 'petal length (cm)', fig = fig)


res = results.resid
fig = sm.qqplot(res, fit=True, line="45")
plt.show()

#MSE
from sklearn.metrics import mean_squared_error
print("MSE:" + str(mean_squared_error(Y_test, y_predicted).round(2)))
