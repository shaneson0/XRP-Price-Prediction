

import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.normalization import BatchNormalization


df=pd.read_csv('./data/xrp.csv', sep=',',header=0)

headres = df.head()
print(headres)
std = StandardScaler()

def spitData(df):
    X = std.fit_transform(df[['Open*', 'High', 'Low', 'Volume', 'Market Cap']])
    y = std.fit_transform(df[['Close']])
    print(y)
    # print(std.inverse_transform(y))
    X_train = X[:700]
    X_test = X[701:]
    y_train = y[:700]
    y_test = y[701:]
    return X_train, y_train, X_test, y_test

def inverseData(std_data):
    return std.inverse_transform(std_data)

def buildModel(batch_size, timesteps, input_dim, neurons):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, timesteps, input_dim)))
    model.compile(loss='mse', optimizer='adam')

    return model

def fitModel(X_train, y_train, nb_epoch, batch_size,  model):
    X_train = X_train.reshape(X_train.shape[0], 1, )
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
    return model


X_train, y_train, X_test, y_test = spitData(df)

batch_size = 1
neurons = 5
model = buildModel(batch_size, X_train.shape[0], X_train.shape[1], neurons)
model = fitModel(X_train, y_train, 100, batch_size, model)










