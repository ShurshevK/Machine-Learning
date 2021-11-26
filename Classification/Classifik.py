import pandas as pd
from sklearn import datasets
import numpy as np
irisdata = datasets.load_iris()
df = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
df["species"] = irisdata.target_names[irisdata.target]
print(df.groupby('species').size())
print (df)
X = df[['petal length (cm)', 'petal width (cm)']]
y = df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1,stratify=y)
# if variable y is a variable with values 0 and  
# 1 and there are 20% of 0 and 80% of 1, "stratify=y"
# will make the random split has 20% of 0 and 80% of 1.

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
#This works because shuffle takes precedence over random_state and random_state is ignored.

print(y_train.value_counts())
print(y_test.value_counts())



from sklearn.neighbors import KNeighborsClassifier
## instantiate 
knn = KNeighborsClassifier(n_neighbors=5)
## fit 
knn.fit(X_train, y_train)
#predict
pred = knn.predict(X_test)
print(pred[:5])
#test
print((pred==y_test.values).sum())
print(y_test.size)

#test percent
print((pred==y_test.values).sum()/y_test.size)
print(knn.score(X_test, y_test))

#confusion matrix
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, pred))
#vizualise conf. matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.GnBu)

#k-fold cross validation
from sklearn.model_selection import cross_val_score
knn_cv = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#tuning the hyperparameter
#grid search
from sklearn.model_selection import GridSearchCV
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
print(knn_gscv.best_score_)

knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)

y_pred = knn_final.predict(X)
print(knn_final.score(X, y))

#We can report that our final model, 4nn, has an accuracy of 97.3% in predicting the species of iris!