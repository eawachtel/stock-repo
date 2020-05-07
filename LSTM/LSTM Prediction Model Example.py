
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import os


start = dt.datetime(2014, 1, 1)
end = dt.datetime(2019, 8, 22)
ticker = 'RJF'
df = web.DataReader(ticker, 'yahoo', start, end)
df.to_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))

df = pd.read_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
origdf = df
df.tail()

# Get Adj Close data
df = df['Adj Close'].values
df = df.reshape(-1, 1)
print(df.shape)
df[:5]

#Split data into training and test data

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8)-50:])
print(dataset_train.shape)
# print(dataset_test.shape)
print(dataset_train[1])
# print(dataset_test[1])

a=plt.figure(1)
plt.plot(dataset_train, linewidth=3, color='red', label='Training')
plt.plot(dataset_test, linewidth=1, color='blue', label='Test')
plt.plot(df, linewidth=1, color='black', label=ticker)
plt.legend(['Training',ticker], loc='upper left')
a.show()


# Scale training data
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:5]

#Scale Test data
dataset_test = scaler.transform(dataset_test)
dataset_test[:5]

testlen = df.shape[0]
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        datashape = df.shape[0]
        test = df[i-50:i, 0]
        length = len(test)
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


x_train, y_train = create_dataset(dataset_train)
x_train[:1]
#
x_test, y_test = create_dataset(dataset_test)
x_test[:1]

# Reshape features for LSTM Layer
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print('building model')
# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=96, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=96, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=96))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# if(not os.path.exists('stock_prediction.h5')):
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('C:/Projects/Stock Tools/Model/stock_prediction.h5')

print('predicting model')
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

print('plotting results')
b=plt.figure(2)
plt.plot(range(len(y_train)+50,len(y_train)+50+len(predictions)),predictions, color='blue',
        label='Predicted Testing Price')
plt.plot(df, color='red',  label="True Price")
b.show()


x = x_test[-1]
num_timesteps = 100
preds = []
for i in range(num_timesteps):
    data = np.expand_dims(x, axis=0)
    prediction = model.predict(data)
    prediction = scaler.inverse_transform(prediction)
    preds.append(prediction[0][0])
    x = np.delete(x, 0, axis=0) # delete first row
    x = np.vstack([x, prediction]) # add prediction

c = plt.figure(3)
plt.plot(dataset_train, linewidth=3, color='red', label='Training')
plt.plot(dataset_test, linewidth=1, color='blue', label='Test')
plt.plot(x, linewidth=1, color='orange', label='Predict')
plt.plot(df, color='red', label="True Price")
c.show()
stop = x




