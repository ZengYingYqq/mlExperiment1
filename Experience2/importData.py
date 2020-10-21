import pandas
import numpy as np
data = pandas.read_csv("data3.0.txt", sep=',')
print(data.shape)
melonWeight = np.array(data['密度'])
melonSugar = np.array(data['含糖率'])
dataList = list(data.values)
dataList = np.array(dataList)

print(dataList.shape)
print(dataList[:, 9])

def count(isGood):
    count = 0
    for i in range(0, len(isGood)):
        if(isGood[i] == '是 '):
            count = count+1
    gailv = count/len(isGood)
    shang = np.log(gailv)*gailv*(-1)-np.log(1-gailv)*(1-gailv)*(-1)
    print(shang)

count(dataList[:, 9])


def opShang(suxings, melons):
    allSuxing = dataList[:, 1:9]
    SuXingShang = []

