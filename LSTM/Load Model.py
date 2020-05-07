import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import os
from numpy import array


start = dt.datetime(2014, 1, 1)
end = dt.datetime(2019, 8, 22)
ticker = 'RJF'
df = web.DataReader(ticker, 'yahoo', start, end)
df.to_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))

df = pd.read_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
df.tail()

# Get Adj Close data
df = df['Adj Close'].values
df = df.reshape(-1, 1)
print(df.shape)
df[:5]

#Split data into training and test data

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])
dataset_test_orig = dataset_test
print(dataset_train.shape)
print(dataset_test.shape)
print(dataset_train[1])
print(dataset_test[1])

a=plt.figure(1)
plt.plot(dataset_train, linewidth=3, color='red', label='Training')
plt.plot(dataset_test, linewidth=1, color='blue', label='Test')
plt.plot(df, linewidth=1, color='black', label=ticker)
plt.legend(['Training','Test',ticker], loc='upper left')
a.show()


# Scale training data
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:5]

# Scale Test data
dataset_test = scaler.transform(dataset_test)
dataset_test[:5]

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


def create_dataset2(df):
    x = []
    y = []
    for i in range(df.shape[0] - 50, df.shape[0]):
        x.append(df[i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

def create_dataset3(df):
    x = []
    y = []
    for i in range(df.shape[0] - 50, df.shape[0]):
        x.append(df[i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_train[:1]

# x_test, y_test = create_dataset2(dataset_test)
# x_test[:1]

x_test, y_test = create_dataset3(dataset_train)
x_test[:1]


# Reshape features for LSTM Layer


# x_test = array(x_test)
# x_test = x_test.reshape((1, x_test.shape[0], 1))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], 1))
# test = x_test.shape[1]
#Load Model
model = tf.keras.models.load_model('C:/Projects/Stock Tools/Model/stock_prediction.h5')

print('predicting model')
days_predicted = len(dataset_test)
predictions_list = np.zeros(shape=(1,1))
for x in range(days_predicted):
    x_test2 = array(x_test)
    x_test3 = x_test2.reshape((1, x_test2.shape[0], 1))
    predictions = model.predict(x_test3)
    next_val = predictions[0][0]
    x_test = x_test[1:]
    x_test = np.append(x_test, next_val)
    predictions_list = np.append(predictions_list, predictions)

predictions_list = predictions_list[1:]
predictions_list = predictions_list.reshape(predictions_list.shape[0],1)
predictions = scaler.inverse_transform(predictions_list)
dataset_train = scaler.inverse_transform(dataset_train)
offsetx =len(dataset_train)
offsetx_list = [offsetx]
for i in range(1, len(predictions)):
    offsetx_list.append(offsetx_list[i-1] + 1)


print('plotting results')
b=plt.figure(2)
plt.plot(offsetx_list, predictions, color="blue",
        label='Predicted Testing Price')
plt.plot(dataset_train, linewidth=1, color='red', label='Training')
plt.plot(offsetx_list, dataset_test_orig, linewidth=1, color='red', label='Test')
b.show()


