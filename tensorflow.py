#install required packages
# !pip install yfinance
# !pip install pandas_ta

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas_datareader import data
import yfinance as yf

#fix random state
tf.random.set_seed(20)
#window size
window_size = 60

start_date = '2021-03-24' #added 100 days to generate 100 days EMA (will drop 100 NAs to bring databset back to 1 year)
end_date = '2022-03-25'

panel_data = yf.download(tickers="VGT", start=start_date, end=end_date)

#Shift close price to index 0
# close_price = panel_data.pop('Close')
# panel_data.insert(0, 'Close', close_price)

#return series
panel_data['PCT_CHNG'] = (panel_data['Adj Close'].pct_change() +1).cumprod() - 1
PCT_CHNG = panel_data.pop('PCT_CHNG')
panel_data.insert(0, 'PCT_CHNG', PCT_CHNG)


#start of function ===========================================

#creating dataframe
data = panel_data.sort_index(ascending=True, axis=0)

data = data.drop(columns=['Close', 'Open', 'Low', 'High', 'Adj Close', 'Volume'])

#Drop NA values
new_data = data.dropna()

dataset = new_data.values

#creating train and test datasets with approximate 80% train 
train_len = (new_data.shape[0] // 10) * 8
train = dataset[:train_len, :]
test = dataset[train_len:, :]

#scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(train)
scaled_data_test = scaler.transform(test)

#scaler for inverse
inverse_scaler = MinMaxScaler(feature_range=(0, 1))
inverse_scaler.fit(train[:,0].reshape(-1, 1)) #close price at position 0

x_train, y_train = [], []

for i in range(window_size,len(scaled_data_train)):
    # for 1 variable
    if scaled_data_train.shape[1] == 1:
      x_train.append(scaled_data_train[i-window_size :i, 0])
    else:
      x_train.append(scaled_data_train[i-window_size :i])
    y_train.append(scaled_data_train[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# for 1 variable
if scaled_data_train.shape[1] == 1:
  x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# print(x_train.shape)

# create and fit the LSTM network
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(units=32))
model.add(Dense(1))

adam=tf.keras.optimizers.Adam(learning_rate=0.02) #0.005
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=2, callbacks=[reduce_lr, callback])

#predicting using past [window size] from the train data
inputs = new_data[len(new_data) - len(test) - window_size:].values
# for 1 variable
if scaled_data_train.shape[1] == 1:
  inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(window_size,inputs.shape[0]):
  # for 1 variable
  if scaled_data_train.shape[1] == 1:
    X_test.append(inputs[i-window_size:i,0])
  else:
    X_test.append(inputs[i-window_size:i])
X_test = np.array(X_test)

# for 1 variable
if scaled_data_train.shape[1] == 1:
  X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = inverse_scaler.inverse_transform(closing_price)

#for plotting
train = new_data[:train_len+1]
test = new_data[train_len:]
test['Predictions'] = closing_price

#align predicted and actual
# print(test)
# if test['Close'][0] != test['Predictions'][0]:
#   if test['Close'][0] > test['Predictions'][0]:
#     test['Predictions'] = test['Predictions'] + (test['Close'][0] - test['Predictions'][0])
#   else:
#     test['Predictions'] = test['Predictions'] - (test['Predictions'][0] - test['Close '][0])
# print(test)
#mse
# rmse = mean_squared_error(list(test['Close']), list(test['Predictions']), squared=False)
rmse = mean_squared_error(list(test['PCT_CHNG']), list(test['Predictions']), squared=False)
print("")
print("RMSE: ", rmse)

# fig, ax = plt.subplots(figsize=(16,9))
# fig.suptitle('Stock Price Prediction', fontsize=16)
# ax.plot(train['PCT_CHNG'], label = 'Return series')
# ax.plot(test['PCT_CHNG'] , label = 'Actual Return series')
# ax.plot(test['Predictions'] , label = 'Predicted Return series')
# ax.legend()
