import pandas
from math import log
from operator import itemgetter


# 计算香农熵
def computerEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    entropy = 0.0
    for key in labelCounts.keys():
        #得到概率
        p_i = float(labelCounts[key] / numEntries)
        #根据公式得到熵
        entropy -= p_i * log(p_i, 2)
    return entropy

# 划分数据集,找出第axis个属性为value的数据
def splitDataSet(dataSet, axis, value):
    returnSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            retVec = featVec[:axis]
            retVec.extend(featVec[axis + 1:])
            returnSet.append(retVec)
    return returnSet

# 根据信息增益原则得到最优属性
def getBestFeat(dataSet):
    numFeat = len(dataSet[0]) - 1
    # 初始的信息熵
    Entropy = computerEntropy(dataSet)
    DataSetlen = float(len(dataSet))
    bestGain = 0.0
    bestFeat = -1
    # 分别根据每个属性划分样本集合
    for i in range(numFeat):
        allvalue = [featVec[i] for featVec in dataSet]
        specvalue = set(allvalue)
        nowEntropy = 0.0
        # 计算划分后各个样本集合的信息熵之和
        for v in specvalue:
            Dv = splitDataSet(dataSet, i, v)
            p = len(Dv) / DataSetlen
            nowEntropy += p * computerEntropy(Dv)
        # 选择出信息增益最大的样本
        if Entropy - nowEntropy > bestGain:
            bestGain = Entropy - nowEntropy
            bestFeat = i
    return bestFeat


def Vote(classList):
    classdic = {}
    for vote in classList:
        if vote not in classdic.keys():
            classdic[vote] = 0
        classdic[vote] += 1
    sortedclassDic = sorted(classdic.items(), key=itemgetter(1), reverse=True)
    return sortedclassDic[0][0]


def createDecisionTree(dataSet, featnames):
    featname = featnames[:]
    # 此节点的分类情况
    classlist = [featvec[-1] for featvec in dataSet]
    # 如果全部属于一类，则该节点为叶节点
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    # 如果该节点没有任何属性进行后续分类了，则根据少数服从多数原则设置为叶节点
    if len(dataSet[0]) == 1:
        return Vote(classlist)
    # 选择一个最优特征划分当前节点的样本集合
    bestFeat = getBestFeat(dataSet)
    bestFeatname = featname[bestFeat]
    del (featname[bestFeat])
    DecisionTree = {bestFeatname: {}}
    # 根据当前特征的属性值数量创建分支
    allvalue = [vec[bestFeat] for vec in dataSet]
    specvalue = set(allvalue)
    for v in specvalue:
        copyfeatname = featname[:]
        DecisionTree[bestFeatname][v] = createDecisionTree(splitDataSet(dataSet, bestFeat, v), copyfeatname)
    return DecisionTree


if __name__ == '__main__':
    # 导入原始数据
    data = pandas.read_csv("data3.0.txt", sep=',')
    # 选择原始数据的部分属性
    temp = data[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜 ']].values
    # 处理原始数据，并重新设置索引
    data1 = pandas.DataFrame(temp)
    data1.columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    dataSet = data1.values.tolist()
    # 初始的特征集合
    featname = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    Tree = createDecisionTree(dataSet, featname)
    print(Tree)

