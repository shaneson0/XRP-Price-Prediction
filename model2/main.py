

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df=pd.read_csv('../data/xrp.csv', sep=',',header=0)

headres = df.head()
# 变成一列

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

Features = np.hstack((OpenCap, HighCap, LowCap, VolumeCap, MarketCap))
# print(Features)

# Split Data
trainX, testX = Features[0: 700, :], Features[700:len(Features) + 1, :]
trainY, testY = CloseCap[0: 700, :], CloseCap[700:len(CloseCap) + 1, :]

print(len(trainX))
print(len(testX))

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# prepare LSTM
neurons = 8
batch_size = 1
epochs = 60

model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#
# for i in range(epochs):
#     model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#     model.reset_states()
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, shuffle=True)

# train_reshaped = trainX[:, 0].reshape(len(trainX), 1, 1)
# forecast the entire training dataset to build up state for forecasting
TestTrainQuality = model.predict(trainX, batch_size=batch_size)
# print(TestTrainQuality)


# predict ALl Text X
TextResults = model.predict(testX, batch_size=batch_size)
TextResults = Scaler.inverse_transform(TextResults)
print(TextResults)
TextTureResult = Scaler.inverse_transform(testY)

# show the result
# Test RMSE: 0.01514
rmse = sqrt(mean_squared_error(TextTureResult, TextResults))
print('Test RMSE: %.5f' % rmse)
# line plot of observed vs predicted// blue is the real value


pyplot.plot(TextTureResult, label='true')
pyplot.plot(TextResults, label='predict')
pyplot.show()































