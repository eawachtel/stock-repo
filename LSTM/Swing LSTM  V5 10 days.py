import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
from numpy import array
from numpy.random import seed
np.random.seed(0)
# from tensorflow import set_random_seed
# set_random_seed(1)
from pandas_datareader import data as pdr
import yfinance as yf


################################################# Variables ###################################################


Load_Saved_Ticker_data = False
ticker = 'RJF'
start = dt.datetime(2018, 3, 1)
end = dt.datetime(2019, 9, 25)
plot_train_test = True
split_method = 1
days_grouped = 60 #60
build_model = True
load_model = False
epoch= [50] #500
batchsize=32
days_predicted = 30
mean_test_error_array = []
mean_prediction_error_array = []

for s in epoch:
    epoch = s
    from numpy.random import seed
    seed(0)
    # from tensorflow import set_random_seed
    # set_random_seed(1)


# Split Method 1 is 70% train / 20% test / 10% predict
################################################################################################################


################################################# Get Stock Data ###############################################

    yf.pdr_override()  # <== that's all it takes :-)
    df = pdr.get_data_yahoo(ticker, period='5d', interval='1m')


    # if Load_Saved_Ticker_data == False:
    #     df = web.DataReader(ticker, 'yahoo', start, end)
    #     df.to_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
    # if Load_Saved_Ticker_data == True:
    #     df = pd.read_csv("C:/Projects/Stock Tools/Stock Data/{ticker}.csv".format(ticker=ticker))
    df = df['Adj Close'].values
    df = df.reshape(-1, 1)
    print('Stock data Loaded')

    ################################################################################################################


    ################################################# Training / Test ##############################################
    if split_method == 1:
        dataset_train = np.array(df[:int(df.shape[0]*0.8)])
        dataset_test = np.array(df[int(df.shape[0]*0.8)-days_grouped:int(df.shape[0]*1)-days_predicted])
        dataset_train_orig = dataset_train
        dataset_test_orig = dataset_test
    dataset_test_y = []
    dataset_test_orig_x = []  # minus 50 to account for day 51 being the first plotted day
    train_len = len(dataset_train)-days_grouped  # Subtract 50 day offset
    test_len = len(dataset_test)

    for i in range(0, test_len):
        var = train_len + i
        var2 = var - days_grouped
        dataset_test_y.append(var)
        dataset_test_orig_x.append(var2)

    print(len(dataset_test_y))
    print(len(dataset_test))

    if plot_train_test == True:
        a=plt.figure(1)
        # plt.plot(df, linewidth=2, color='black', label=ticker)  ### Plot orig ticker data base layer if needed
        plt.plot(dataset_train, linewidth=1, color='red', label='Training')
        plt.plot(dataset_test_y, dataset_test, linewidth=1, color='blue', label='Test')
        plt.legend(['Training','Test',ticker], loc='upper left')
        a.show()

    ################################################################################################################



    # Scale training data
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_train[:5]

    #Scale Test data
    dataset_test = scaler.transform(dataset_test)
    dataset_test[:5]

    def create_dataset(df):
        x = []
        y = []
        for i in range(days_grouped, df.shape[0]):
            x.append(df[i-days_grouped:i, 0])
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

    if build_model == True:
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
        # history = model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize)
        model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize, shuffle=False)

        # model.save('C:/Projects/Stock Tools/Model/stock_prediction_' + ticker + '_start_' + start.strftime('%m%d%Y')
        #            + '_end_' + end.strftime('%m%d%Y') + '_daysgrouped_' + str(days_grouped) + '.h5')

    if load_model == True:
        model = tf.keras.models.load_model('C:/Projects/Stock Tools/Model/stock_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    print('plotting test results')
    b=plt.figure(2)
    plt.plot(range(len(y_train),(len(y_train)+len(predictions))), predictions, color='blue',
            label='Predicted Testing Price')
    plt.plot(df[days_grouped:int(df.shape[0]*0.7)], color='red',  label="True Price")
    plt.plot(dataset_test_orig_x, dataset_test_orig, linewidth=1, color='black', label='True Test')
    # plt.plot(df, color='black',  label="True Price")
    b.show()

    # Test Error
    test_data_orig_mod = dataset_test_orig[days_grouped:int(dataset_test_orig.shape[0]*1)]
    test_error = ((predictions - test_data_orig_mod) / test_data_orig_mod) * 100
    z=plt.figure(3)
    plt.plot(test_error, color='black',  label="Test Error")
    plt.legend(['Test Error', ticker], loc='upper left')
    z.show()

    print('predicting model')


    ####################################### Predict last 10% of test ##########################################

    def create_dataset3(df):
        x = []
        y = []
        for i in range(df.shape[0] - days_grouped, df.shape[0]):
            x.append(df[i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y


    dataset_proj = np.array(df[int(df.shape[0]*1)-days_grouped-days_predicted:int(df.shape[0]*1)-days_predicted])
    data_proj_raw = df[int(df.shape[0]*1)-days_predicted:int(df.shape[0]*1)]
    dataset_proj = scaler.transform(dataset_proj)
    x_proj, y_proj = create_dataset3(dataset_proj)

    print('predicting model')

    predictions_list = np.zeros(shape=(1,1))
    for x in range(days_predicted):
        x_test2 = array(x_proj)
        x_test3 = x_test2.reshape((1, x_test2.shape[0], 1))
        predictions = model.predict(x_test3)
        next_val = predictions[0][0]
        x_proj = x_proj[1:]
        x_proj = np.append(x_proj, next_val)
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
    c = plt.figure(4)
    plt.plot(predictions, color="blue", label='Proj Model')
    plt.plot(data_proj_raw, color="black", label='Proj Actual')
    c.show()

    # Test Error
    test_data_orig_mod = dataset_test_orig[days_grouped:int(dataset_test_orig.shape[0]*1)]
    prediction_error = ((predictions - data_proj_raw) / data_proj_raw) * 100
    y=plt.figure(6)
    plt.plot(prediction_error, color='black',  label="Prediction Error")
    plt.legend(['Prediction Error', ticker], loc='upper left')
    y.show()

    mean_prediction_error = round(np.mean(prediction_error), 4)
    mean_test_error = round(np.mean(test_error), 4)

    mean_test_error_array.append(mean_test_error)
    mean_prediction_error_array.append(mean_prediction_error)
    del model
    del data_proj_raw
    del dataset_proj
    del dataset_test
    del dataset_test_orig
    del dataset_test_orig_x
    del dataset_test_y
    del dataset_train
    del dataset_train_orig
    del df
    del mean_prediction_error
    # del mean_prediction_error_array
    del mean_test_error
    # del mean_test_error_array
    del next_val
    del offsetx
    del offsetx_list
    del prediction_error
    del predictions
    del predictions_list
    del test_data_orig_mod
    del test_error
    del train_len
    del var
    del var2
    del x_proj
    del x_test
    del x_test2
    del x_test3
    del x_train
    del y_proj
    del y_test
    del y_train

    print(mean_test_error_array)
    print(mean_prediction_error_array)






