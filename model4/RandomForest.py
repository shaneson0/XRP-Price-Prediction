# /usr/bin/python

# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json

#提取数据&数据预处理
data = pd.read_csv('\\xrp.csv', parse_dates=[0], index_col=0)
data = data[::-1]
x1 = pd.bdate_range(start='2016-9-24', periods=760, freq='D')
x = pd.bdate_range(start='2018-8-25', periods=60, freq='D')
X = data['Open']
X_test = []
for i in range(700, 760):
    X_test.append(X[i])
X_train = []
for i in range(700):
    X_train.append(X[i])
y = data['Close']
y_test = []
for i in range(700, 760):
    y_test.append(y[i])
y_train =[]
for i in range(700):
   y_train.append(y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y = np.array(y)
#随机深林回归
rfr =RandomForestRegressor(n_estimators=500)
rfr.fit(X_train.reshape(-1,1), y_train.ravel())
y_pre =rfr.predict(X_test.reshape(-1,1))
print("60天的实际值为：")
print(y_test)
print("随机森林预测xrp60天的值为：")
print(y_pre)
#data ={}
#data["Date"] =[i for i in range(1,61)]
#data["Predict Close"] = y_pre.tolist()
#jsonStr = json.dumps(data)
#fileObject = open('Predict data.json', 'w')
#fileObject.write(jsonStr)
#fileObject.close()
print("test RMSE:%.3f"%mean_squared_error(y_test, y_pre)**0.5)

#结果可视化

plt.plot(x, y_test, 'blue', linewidth =0.5, label="Market Price")
plt.plot(x, y_pre, 'red', linewidth=0.5, label="Predicted Price")
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('value')
plt.title("Predicted and market price of xrp latest 60 days")
plt.show()


