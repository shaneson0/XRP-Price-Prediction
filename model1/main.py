

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df=pd.read_csv('../data/xrp.csv', sep=',',header=0)

headres = df.head()

#          Date     Open*      High       Low     Close       Volume    Market Cap
# 0  2018/10/23  0.453070  0.471596  0.439644  0.462058  445238000.0  9.418537e+08
# 1  2018/10/22  0.456089  0.458188  0.450583  0.453109  249302000.0  1.062607e+09
# 2  2018/10/21  0.458863  0.463629  0.455411  0.456694  262867000.0  1.173547e+09
# 3  2018/10/20  0.453936  0.461899  0.451070  0.459151  277061000.0  9.764962e+08
# 4  2018/10/19  0.457344  0.460625  0.450369  0.454178  303401000.0  1.112828e+09
# print(headres)

# 变成一列
values = df['Close'].values.reshape(-1,1)
values = values.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# print(values)


# Split Data
train, test = scaled[0: 700, :], scaled[701:len(scaled),:]

# print(len(train))
# print(len(test))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

batch_size = 5
epochs = 250

model = Sequential()
model.add(LSTM(batch_size, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0, shuffle=False)


# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

yhat = model.predict(testX)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))


pyplot.plot(yhat_inverse, label='predict')
pyplot.plot(testY_inverse, label='true')
pyplot.legend()
pyplot.show()

# RMSE = 0.001
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)




