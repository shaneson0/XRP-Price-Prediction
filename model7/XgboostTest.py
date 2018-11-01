

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ------- Data Prepare -------

df = pd.read_csv('../data/xrp.csv', sep=',',header=0)

Scaler = MinMaxScaler(feature_range=(0, 1))

CloseCap = Scaler.fit_transform(df['Close'].values.reshape(-1, 1))
OpenCap = Scaler.fit_transform(df['Open*'].values.reshape(-1, 1))
Dvalue = Scaler.fit_transform((df['Close'] - df['Open*']).values.reshape(-1, 1))
VolumeCap = Scaler.fit_transform(df['Volume'].values.reshape(-1, 1))

RawDatas = np.hstack((CloseCap, Dvalue, VolumeCap))

shift = 5

def process_price(X, shift):
    x,y = [], []
    length = len(X) - shift
    for i in range(length):
        x.append(X[i:shift+i, :])
        y.append(X[shift+i, 0])
    return np.array(x), np.array(y)

x, y = process_price(RawDatas, shift)
y = y.reshape(-1,1)

print(x.shape)

# x = xgb.DMatrix(x)
# y = xgb.DMatrix(y)
#
# trainX, testX = x[0: 700], x[700:len(x) + 1]
# trainY, testY = y[0: 700], y[700:len(y) + 1]
#
# print(trainX)




# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
#
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
#
# # make prediction
# preds = bst.predict(dtest)

