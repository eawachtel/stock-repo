import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import sqlite3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')

ticker = 'AAPL'


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
except Exception:
    print('[!] Could not connect to DB')

#Create new dataframe with close - open (candlestick)
df['Candle'] = 0
df['Candle'] = df['Close'] - df['Open']
data = df.filter(['Candle'])

#Visualize stockdata
plt.figure(figsize=(16,8))
plt.title('Price Delta History')
plt.plot(df['Candle'])
plt.show()

#Conver the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

print(training_data_len)
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#Visualize scaleddata
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(scaled_data)
# plt.show()

#Create the training data set
#Create the scaled training data set
training_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train sets
x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)
#Reshape the data
#LSTM expects [3D array] number of samples, number of time steps, number of features
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile model
model.compile(optimizer='adam', loss='mean_squared_error')
rmse = 5
while rmse > .5:
    #Train model, batch_size is total # of training samples in a batch, epochs is number of iteration passed through model
    model.fit(x_train, y_train, batch_size=15, epochs=10)

    #Create the testing data set
    #Create a new array containing scaled values 1543 to 2003 (scaled testing dataset)
    test_data = scaled_data[training_data_len-60:, :]
    #Create the datasets x_test, y_test

    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    #Convert data to numpy array
    x_test = np.array(x_test)
    #Reshape data to 3D for LSTM input
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Get the root mean squared error (RSME)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# #Visualize Data
# plt.figure(figsize=(16, 8))
# plt.title('Model')
# plt. xlabel('Date', fontsize= 18)
# plt.ylabel('Close Price USD', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

#show the valid and predicted prices
print(valid)

pred_price_list = []
predictedIndex = []
days_predicted = 15
for i in range(0, days_predicted):

    predictedIndex.append(i)

    last_60_days = df.Candle.tail(60 - i).values
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

max = df.index.max()

for i in range(0, len(predictedIndex)):
    predictedIndex[i] = predictedIndex[i] + 1 + max

fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Candle'], name='Current Price',
                                 line=dict(color='red', width=4)),
                      go.Scatter(x=predictedIndex, y=pred_price_list, name='Predictions',
                                 line=dict(color='blue', width=4))

                      ])

fig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    xaxis_showgrid=False,
    yaxis_showgrid=False
)
fig.show()


#Scale the data to the values between 0 1
# last_60_days_scaled = scaler.transform(last_60_days)
# X_test = []
#Append the past 60 days
# X_test.append(last_60_days_scaled)4



debugger = 'debugger'