import math
import numpy as np
from numpy import array
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import sqlite3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')

ticker = 'AAPL'
trainModel = True

conn = sqlite3.connect('C:/sqlite/stock-data.db')
try:
    cur = conn.cursor()
    sql = "select * FROM price where Ticker = '{ticker}'".format(ticker=ticker)
    cur.execute(sql)
    col_name_list = [tuple[0] for tuple in cur.description]
    rows = cur.fetchall()
    stockDataDF = pd.DataFrame(rows)
    stockDataDF.columns = col_name_list
    df = stockDataDF
    dfTest = df.head(350)
except Exception:
    print('[!] Could not connect to DB')

#Visualize stockdata
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.show()

#Create new dataframe with only the 'Close' column
# data = df.filter(['Close'])
Close = dfTest.filter(['Close'])
Close['Close'] = Close['Close'].rolling(window=5).mean()
Close = Close[20:]

RSI = dfTest.filter(['RSI'])
RSI['RSI'] = RSI['RSI']
RSI = RSI[20:]

#Conver the dataframe to a numpy array
closeDataset = Close.values
RSIDataset = RSI.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(closeDataset) * .8)

print(training_data_len)
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
# closeScaledData = scaler.fit_transform(closeDataset)
# rsiScaledData = scaler.fit_transform(RSIDataset)
closeScaledData = closeDataset
rsiScaledData = RSIDataset


#Visualize scaleddata
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(scaled_data)
# plt.show()

if trainModel:
    #Create the training data set
    #Create the scaled training data set
    closeScaledDataTrain = closeScaledData[0:training_data_len, :]
    rsiScaledDataTrain = rsiScaledData[0:training_data_len, :]
    #Split the data into x_train and y_train sets

    x_train = []
    y_train = []

    for i in range(60, len(closeScaledDataTrain)):
        xclose = closeScaledDataTrain[i-60:i, 0]
        xrsi = rsiScaledDataTrain[i-60:i, 0]
        y_train.append(closeScaledDataTrain[i, 0])
        x_train.append([xclose, xrsi])
        # x_train_append = []
        # for z in range(0, len(xclose)):
        #     xcloseappend = xclose[z]
        #     xrsiappend = xrsi[z]
        #     x_train_append.append([xcloseappend, xrsiappend])
        # x_train.append(x_train_append)


    x_train_array = array(x_train)
    print(x_train_array.shape[0])
    x_train = x_train_array.reshape(x_train_array.shape[0], 60, 2)

    #Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    print('x train[0]', x_train.shape[0])
    print('x train[1]', x_train.shape[1])
    #Reshape the data
    #LSTM expects [3D array] number of samples, number of time steps, number of features
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    #Build LSTM model
    model = Sequential()
    model.add(LSTM(5, return_sequences=True, input_shape=(60, 2)))
    model.add(LSTM(5, return_sequences=False))
    # # model.add(Dropout(0.2))
    # # model.add(Dense(25))
    model.add(Dense(1))

    #Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.save('C:/stock-repo/model-file-' + ticker + '.h5')

else:
    model = load_model('C:/stock-repo/model-file-' + ticker + '.h5')

if trainModel:
    rmse = 5
    while rmse > .5:
        #Train model, batch_size is total # of training samples in a batch, epochs is number of iteration passed through model
        model.fit(x_train, y_train, batch_size=10, epochs=5, verbose=2)

        #Create the testing data set
        #Create a new array containing scaled values 1543 to 2003 (scaled testing dataset)
        test_len = (training_data_len - 60)
        test_data = closeScaledData[test_len:, :]

        #Create the datasets x_test, y_test
        closeScaledDataTest = closeScaledData[training_data_len - 60:, :]
        rsiScaledDataTest = rsiScaledData[training_data_len - 60:, :]
        x_test = []
        y_test = closeDataset[training_data_len:, :]

        print(len(test_data))
        print(len(closeScaledDataTest))
        print(len(rsiScaledDataTest))

        for i in range(60, len(test_data)):
            xclose = closeScaledDataTest[i - 60:i, 0]
            xrsi = rsiScaledDataTest[i - 60:i, 0]

            x_test_append = []
            for z in range(0, len(xclose)):
                xcloseappend = xclose[z]
                xrsiappend = xrsi[z]
                x_test_append.append([xcloseappend, xrsiappend])
            x_test.append(x_test_append)

        #Convert data to numpy array
        x_test = np.array(x_test)
        #Reshape data to 3D for LSTM input
        x_test = np.reshape(x_test, (x_test.shape[0], 60, 2))

        #Get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        #Get the root mean squared error (RSME)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        print(rmse)

# #Plot the data
# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid['Predictions'] = predictions

pred_price_list = []
predictedIndex = []
days_predicted = 15
for i in range(0, days_predicted):

    predictedIndex.append(i)

    last_60_days = dfTest.Close.tail(60 - i).values
    if len(pred_price_list) > 0:
        for p in pred_price_list:
            last_60_days = np.append(last_60_days, p)
    last_60_days = np.reshape(last_60_days, (last_60_days.shape[0], 1))
    last_60_days = scaler.fit_transform(last_60_days)

    X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))


    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #Undo scaling
    pred_price=scaler.inverse_transform(pred_price)
    pred_price_list.append(pred_price[0][0])

print(pred_price_list)

max = dfTest.index.max()

for i in range(0, len(predictedIndex)):
    predictedIndex[i] = predictedIndex[i] + 1 + max

# #Visualize Data
plt.figure(figsize=(16, 8))
plt.title(ticker)
plt. xlabel('Date', fontsize= 18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(df.index, df['Close'])
plt.plot(predictedIndex, pred_price_list)
plt.legend(['Actual Price', 'Predictions'], loc='lower right')
plt.show()

