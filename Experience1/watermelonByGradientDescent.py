import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_table('watermelonData.txt', delimiter=',')
x_train = data.iloc[:, 1:3]
y_train = data.iloc[:, 3:]
x_train = np.mat(x_train)
y_train = np.mat(y_train)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# target（种类）数量
target_num = 2


def gradAscent(dataList, labelList):
    # x数据转化为矩阵
    dataMat = np.mat(dataList)
    # y数据转换为矩阵并转置为列向量
    labelMat = np.mat(labelList).copy()
    # 返回训练矩阵的大小
    m, n = np.shape(dataMat)
    dataMat = np.hstack((dataMat, np.array((np.ones((m, 1))))))
    n = n + 1
    # 步长
    alpha = 0.005
    # 迭代次数
    maxCycles = 10000
    # 权重列向量，初始值设为0
    weights = np.ones((n, 1))
    # 梯度下降算法
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = h - labelMat
        weights = weights - alpha * dataMat.T * error
    return weights


def test(x_train, y_train, x_test, y_test):
    # 返回矩阵的大小
    m, n = np.shape(x_train)
    n = n + 1
    weights = gradAscent(x_train, y_train)

    print(weights)
    m_test, n_test = np.shape(x_test)
    x_testl = np.hstack((x_test, np.array((np.ones((m_test, 1))))))

    cnt = 0
    test_label = np.zeros(m_test)
    for i in range(m_test):
        test_label[i] = sigmoid(np.dot(x_testl[i], weights))
        if (test_label[i] >= 0.5):
            test_label[i] = 1
        else:
            test_label[i] = 0

    for i in range(m_test):
        if y_test[i] == test_label[i]:
            cnt += 1
    print("正确率 = " + str(cnt / m_test))


test(x_train, y_train, x_test, y_test)
