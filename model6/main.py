


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot


df = pd.read_csv('../data/xrp.csv', sep=',',header=0)

# -------------------------------------------------- 数据预处理 ----------------------------------------------------
# 归一化
Scaler = MinMaxScaler(feature_range=(0, 1))
MarketCap = Scaler.fit_transform(df['Market Cap'].values.reshape(-1, 1))
# print(MarketCap)

VolumeCap = Scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
# print(VolumeCap)

LowCap = Scaler.fit_transform(df['Low'].values.reshape(-1, 1))
# print(LowCap)

HighCap = Scaler.fit_transform(df['High'].values.reshape(-1, 1))
# print(HighCap)

OpenCap = Scaler.fit_transform(df['Open*'].values.reshape(-1, 1))
# print(OpenCap)

CloseCap = Scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 差值
Dvalue = Scaler.fit_transform((df['Close'] - df['Open*']).values.reshape(-1, 1))

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

# print(x.shape)
# print(y.shape)
# split train data and test data
trainX, testX = x[0: 700], x[700:len(x) + 1]
trainY, testY = y[0: 700], y[700:len(y) + 1]


# print(trainX.shape)
# print(trainY.shape)


# ------------------------------- Define and Predict LSTM's model -------------------------------

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=10, input_shape=(shift, 3), return_sequences=False))
# model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')

epochs = 120
model.fit(trainX, trainY, epochs = epochs, batch_size = 5, shuffle = True)


# ------------------------------- Test for LSTM's model -------------------------------

prediction = model.predict(testX)
prediction = Scaler.inverse_transform(prediction)

testY = Scaler.inverse_transform(testY)
# calculate RMSE of the predicted close price from norm_y_test
RMSE = math.sqrt(mean_squared_error(testY, prediction))

# Test RMSE: 0.03642
# 0.04596
print('Test RMSE: %.5f' % RMSE)

# print('---- test Y ----')
# print(testY)
# print('---- prediction ----')
# print(prediction)

print(testY.shape)
print(prediction.shape)

pyplot.plot(testY, label='true')
pyplot.plot(prediction, label='predict')
pyplot.show()




























