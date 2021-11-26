import pandas as pd
from sklearn import datasets
import numpy as np
irisdata = datasets.load_iris()
df = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
df["species"] = irisdata.target_names[irisdata.target]

X = df[['petal length (cm)', 'petal width (cm)']]
Y = df['sepal length (cm)']

#Model Overview
import statsmodels.formula.api as smf
import statsmodels.api as sm
X = sm.add_constant(X)

# Fit regression model (using the natural log of one of the regressors)
results = sm.OLS(Y, X).fit()

# Inspect the results
print(results.summary())
#define figure size
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))

#produce residual plots
fig = sm.graphics.plot_regress_exog(results, 'petal width (cm)', fig = fig)

#define residuals
res = results.resid

#create Q-Q plot
fig = sm.qqplot(res, fit=True, line="45")

