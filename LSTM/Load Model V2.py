import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
from numpy import array

################################################# Variables ###################################################

Load_Model = False
Load_Saved_Ticker_data = False
ticker = 'TSLA'
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2019, 8, 22)
split_method = 1
days_grouped = 50

# Split Method 1 is 70% train / 20% test / 10% predict
################################################################################################################


################################################# Get Stock Data ###############################################

if Load_Saved_Ticker_data == False:
    df = web.DataReader(ticker, 'yahoo', start, end)
df.to_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
if Load_Saved_Ticker_data == True:
    df = pd.read_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
df = df['Adj Close'].values
df = df.reshape(-1, 1)
print('Stock data Loaded')

################################################################################################################


################################################# Training / Test ##############################################
if split_method == 1:
    dataset_train = np.array(df[:int(df.shape[0]*0.7)])
    dataset_test = np.array(df[int(df.shape[0]*0.7)-2:int(df.shape[0]*0.9)])
    dataset_test_orig = dataset_train
    dataset_test_orig = dataset_test
dataset_test_y = []
train_len = len(dataset_train)
test_len = len(dataset_test)
for i in range(0, test_len):
    var = train_len + i
    dataset_test_y.append(var)

a=plt.figure(1)
# plt.plot(df, linewidth=2, color='black', label=ticker)  ### Plot orig ticker data base layer if needed
plt.plot(dataset_train, linewidth=1, color='red', label='Training')
plt.plot(dataset_test_y, dataset_test, linewidth=1, color='blue', label='Test')
plt.legend(['Training','Test',ticker], loc='upper left')
a.show()

################################################################################################################


######################################## Scale datasets and prepare data########################################

scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

def create_dataset(df):
    x = []
    y = []
    for i in range(days_grouped, df.shape[0]):
        x.append(df[i-days_grouped:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


def create_dataset2(df):
    x = []
    y = []
    for i in range(df.shape[0] - days_grouped, df.shape[0]):
        x.append(df[i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

def create_dataset3(df):
    x = []
    y = []
    for i in range(df.shape[0] - days_grouped, df.shape[0]):
        x.append(df[i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

if split_method == 1:
    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)
    x_proj, y_proj = create_dataset3(dataset_test)


# Reshape features for LSTM Layer


x_test = array(x_test)
x_test = x_test.reshape((1, x_test.shape[0], 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], 1))
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


