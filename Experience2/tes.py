import pandas
import numpy as np
data = pandas.read_csv("data3.0.txt", sep=',')
print(data.shape)
melonWeight = np.array(data['密度'])
melonSugar = np.array(data['含糖率'])
dataList = list(data.values)
dataList = np.array(dataList)

print(data.columns)
print(data.keys)
