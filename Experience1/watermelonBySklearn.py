from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 数据读取
data = pd.read_table('watermelonData.txt', delimiter=',')
x_train = data.iloc[:, 1:3]
y_train = data.iloc[:, 3:]
x_train = np.mat(x_train)
y_train = np.transpose(np.array(y_train))[0]
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
# 调用模型求解
regr = LogisticRegression()
regr.fit(x_train, y_train)
print('Coefficients :%s,intercept %s' % (regr.coef_, regr.intercept_))
print('Score : %.2f' % regr.score(x_test, y_test))
print(classification_report(y_test, regr.predict(x_test)))
