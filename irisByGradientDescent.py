import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


target_num = 3
iris = load_iris()
# 训练集x（train_size*feature_num）
# 训练集标记y
x_train = iris.data
y_train = iris.target
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def gradAscent(dataList, labelList, t):
    # x数据转化为矩阵
    dataMat = np.mat(dataList)
    # y数据转换为矩阵并转置为列向量
    labelMat = np.mat(labelList).copy()
    labelMat = labelMat.T
    # 返回矩阵的大小
    m, n = np.shape(dataMat)
    dataMat = np.hstack((dataMat, np.array((np.ones((m, 1))))))
    n = n + 1
    for i in range(0, m):
        if labelList[i] == t:
            labelMat[i] = 1
        else:
            labelMat[i] = 0
    # 步长
    alpha = 0.005
    # 迭代次数
    maxCycles = 10000
    # 权重
    weights = np.ones((n, 1))
    # 梯度下降算法
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = h - labelMat
        weights = weights - alpha * dataMat.T * error
    return weights.T


def test(x_train, y_train, x_test, y_test):
    # 返回矩阵的大小
    m, n = np.shape(x_train)
    n = n + 1
    weights = np.mat(np.ones((target_num, n)))
    for i in range(0, target_num):

        weights[i] = gradAscent(x_train, y_train, i)

    print(weights)
    m_test, n_test = np.shape(x_test)
    x_testl = np.hstack((x_test, np.array((np.ones((m_test, 1))))))
    cnt = 0
    test_label = np.zeros(m_test)
    for i in range(m_test):
        for j in range(target_num):
            y = sigmoid(np.dot(weights[j], x_testl[i]))
            if y >= 0.5:
                test_label[i] = j
                break
    for i in range(m_test):
        if y_test[i] == test_label[i]:
            cnt += 1
    print("正确率 = " + str(cnt / m_test))


test(x_train, y_train, x_test, y_test)
