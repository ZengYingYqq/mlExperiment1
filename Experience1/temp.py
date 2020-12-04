
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
X_train = iris.data
y_train = iris.target
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
print(y_test)
regr = LogisticRegression()
regr.fit(X_train, y_train)
print('Coefficients :%s,intercept %s'%(regr.coef_, regr.intercept_))
print('Score : %.2f'%regr.score(X_test, y_test))
print(classification_report(y_test, regr.predict(X_test)))