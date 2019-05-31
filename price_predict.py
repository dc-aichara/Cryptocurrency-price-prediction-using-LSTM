import sys
from PriceIndices import price
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True)
session = tf.Session(config=config)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

price = price()


def price_predict(coin, start_date, end_date):

    print('PYour inputs are {0}, {1}, and {2}'.format(coin, start_date, end_date))
    df = price.get_price(str(coin), str(start_date), str(end_date))

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by ='date')

    df = df.set_index(df['date'])[['price']]

    print('Price data of {0} days have been extracted.'.format(len(df)))

    dataset = df.values

    length = round(len(df)*0.80)

    train = dataset[0:length, :]
    valid = dataset[length:, :]

    # Converting dataset into x_train and y_train

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i, 0]) # Create len(train)-60 batches. Each batch has 60 values
        y_train.append(scaled_data[i, 0])      # one batch of len(train)-60 values
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

    # predicting len(df)-length values, using past 60 from the train data
    inputs = df[len(df) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0]) # len(df)-length batches with each 60 values
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    # Root Mean Square
    rms = np.sqrt(np.mean(np.power((valid-pred_price),2)))

    print(rms)

    # Let's plot results and actual price data

    fig, ax = plt.subplots(figsize =(16, 12))
    train = df[:length]
    valid = df[length:]
    valid['Predictions'] = pred_price
    plt.plot(train['price'], label = 'Price')
    plt.plot(valid[['price']], label = 'Actual Price')
    plt.plot(valid[['Predictions']], color = 'r', label = 'Predicted Price')
    plt.xlabel('Time', color = 'b', fontsize = 24)
    plt.ylabel('{} Price ($)'.format(str(coin).capitalize()), color ='b', fontsize =24)
    plt.title('{0} Price Prediction using LSTM'.format(str(coin).capitalize()), color ='b', fontsize =27)
    plt.grid()
    fig.set_facecolor('orange')
    plt.legend(fontsize =24)
    plt.savefig('results/lstm_price_{0}.png'.format(str(coin)), bbox_inches ='tight', facecolor ='orange')
    plt.show()


if __name__ == '__main__':
    price_predict(sys.argv[1], sys.argv[2], sys.argv[3])

