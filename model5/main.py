

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import math

shift = 5

def process_price(dt, shift):
    x,y = [], []
    length = len(dt) - shift
    for i in range(length):
        x.append(dt[i:shift+i, 0])
        y.append(dt[shift+i, 0])
    return np.array(x), np.array(y)

#generate date dataset
def process_date(dt,shift):
    time_stamp = []
    length = len(dt) - shift
    for i in range(length):
        time_stamp.append(dt[shift+i,0])
    return np.array(time_stamp)


df = pd.read_csv('../data/xrp.csv', sep=',',header=0)

XRP = df[['Date', 'Close']]
HeadInfo = XRP.head()


close_price = XRP['Close'].values
close_price = np.reshape(close_price, (-1,1))

date = XRP['Date'].values
date = np.reshape(date, (-1,1))
newdate = process_date(date, shift)

x, y = process_price(close_price, shift)

# Split Data
trainX, testX = x[0: 700], x[700:len(x) + 1]
trainY, testY = y[0: 700], y[700:len(y) + 1]

# reshape x and y to 2D array before normalization

trainX = np.reshape(trainX, (-1,1))
testX = np.reshape(testX, (-1,1))

trainY = np.reshape(trainY, (-1,1))
testY = np.reshape(testY, (-1,1))

# 字段归一化

scaler = MinMaxScaler()
scaler.fit(trainX)

scaler.fit(trainX)
norm_x_train = scaler.transform(trainX)
norm_x_train = np.reshape(norm_x_train, (-1, shift, 1))

scaler.fit(testX)
norm_x_test = scaler.transform(testX)
norm_x_test = np.reshape(norm_x_test, (-1, shift, 1))


scaler.fit(trainY)
norm_y_train = scaler.transform(trainY)
norm_y_train = np.reshape(norm_y_train, (-1, 1, 1))
norm_y_train = np.reshape(norm_y_train, -1)

scaler.fit(testY)
norm_y_test = scaler.transform(testY)


# LSTM Model

model = Sequential()
model.add(LSTM(units=128, input_shape=(shift, 1), return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(norm_y_train)

model.fit(norm_x_train, norm_y_train, epochs = 100, batch_size = 10, shuffle = True)

# feed norm_x_test dataset to predict norm_y_test
prediction = model.predict(norm_x_test)
prediction = scaler.inverse_transform(prediction)

print(prediction.shape)

# calculate RMSE of the predicted close price from norm_y_test
RMSE = math.sqrt(mean_squared_error(testY, prediction))

# Test RMSE: 0.03142
print('Test RMSE: %.5f' % RMSE)


pyplot.plot(testY, label='true')
pyplot.plot(prediction, label='predict')
pyplot.show()






































