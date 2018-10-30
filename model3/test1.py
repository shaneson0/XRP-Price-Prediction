#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
import  sklearn
def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df


def featureSet(data):
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['Open'])
        tmp_list.append(data.iloc[row]['High'])
        tmp_list.append(data.iloc[row]['Low'])
        tmp_list.append(data.iloc[row]['Close'])
        XList.append(tmp_list)
    yList = data.Close.values
    return XList, yList

'''
Get the last 60 columns of data
'''
def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    XList = []
    for row in range(0, 59):
        tmp_list = []
        tmp_list.append(data.iloc[row]['Open'])
        tmp_list.append(data.iloc[row]['High'])
        tmp_list.append(data.iloc[row]['Low'])
        tmp_list.append(data.iloc[row]['Close'])
        XList.append(tmp_list)
    return XList


'''

获取最后60行的close数据，方便预测
'''
def  get_first60_Truedata(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    trueList = []
    for row in range(0, 59):

        #tmp_list.append(data.iloc[row]['Close'])
        trueList.append(data.iloc[row]['Close'])
    trueData=np.array(trueList)
    return trueData


'''
测试集
'''
def trainandTest(X_train, y_train, X_test,max_depth=5,learning_rate=0.1, n_estimators=160,):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)
    # 对测试集进行预测
    global  ans #######申明为全局变量。这样方便使用。
    ans = model.predict(X_test)
    ans_len = len(ans)
    id_list = np.arange(0, 760)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)
    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'predict'])
    # print(pd_data)
    print ("写入文件")
    pd_data.to_csv('submit.csv', index=None)
    return ans

#########

'''
实现传递参数的方法。
'''

'''
测试集



为了确定boosting 参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：
1、max_depth = 5 :这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。
2、min_child_weight = 1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小。
3、gamma = 0: 起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。
4、subsample,colsample_bytree = 0.8: 这个是最常见的初始值了。典型值的范围在0.5-0.9之间。
5、scale_pos_weight = 1: 这个值是因为类别十分不平衡。
注意哦，上面这些参数的值只是一个初始的估计值，后继需要调优。这里把学习速率就设成默认的0.1。然后用xgboost中的cv函数来确定最佳的决策树数量。前文中的函数可以完成这个工作。
'''
def trainandTestNeedPrams(X_train, y_train, X_test,max_depth=5,learning_rate=0.1, n_estimators=160,):
    # XGBoost训练过程
    #model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False,scale_pos_weight=1, objective='reg:gamma')
    model = xgb.XGBRegressor(max_depth=5,
                             learning_rate=0.1,
                             n_estimators=1000,
                             silent=False,
                             scale_pos_weight=1,
                             objective='multi:softmax')
    model.fit(X_train, y_train)
    # 对测试集进行预测
    global  ans #######申明为全局变量。这样方便使用。
    ans = model.predict(X_test)
    ans_len = len(ans)
    id_list = np.arange(0, 760)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)
    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'predict'])
    # print(pd_data)
    print ("写入文件")
    pd_data.to_csv('submit.csv', index=None)
    return ans


'''
求RMSE
'''
def computerRMSE(target,prediction):
    #target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
    #prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    #print("Errors: ", error)
    #print(error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

   # print("Square Error: ", squaredError)
   # print("Absolute Value of Error: ", absError)

    #print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

    from math import sqrt

    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE


def  mains():
    trainFilePath = 'dataset/soccer/xrp.csv'
    testFilePath = 'dataset/soccer/xrp.csv'
    data = loadDataset(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    predict=trainandTest(X_train, y_train, X_test)
    trueData=get_first60_Truedata('dataset/soccer/xrp.csv')
    computerRMSE(trueData,predict)

    plt.plot(trueData, 'blue',  label="Market Price")
    plt.plot(predict, 'red', label="Predicted Price")
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('value')
    plt.title("Predicted and market price of xrp latest 60 days")
    plt.show()


if __name__ == '__main__':
    # trainFilePath = 'dataset/soccer/xrp.csv'
    # testFilePath = 'dataset/soccer/xrp.csv'
    # data = loadDataset(trainFilePath)
    # X_train, y_train = featureSet(data)
    # X_test = loadTestData(testFilePath)
    # trainandTest(X_train, y_train, X_test)
    mains()
