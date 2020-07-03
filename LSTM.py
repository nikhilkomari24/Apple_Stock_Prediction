"""lstm_basic.py: Predicting stock prices using plain un-tuned LSTM"""
__authors__ = "Keith Herbert, Rasphinder Jagait, Nikhil Komari, Sai Bhasker Raju"
__copyright__ = "Copyright 2019, Comparative Analysis of Currency Prediction Methods"
__license__ = "GPL"
__version__ = "1.0.0"
__maintained_by__ = "Keith Herbert, Rasphinder Jagait, Nikhil Komari, Sai Bhasker Raju"
__email__ = "kherbe@uwo.ca, rjagait@uwo.ca, nkomari@uwo.ca, schittap@uwo.ca"
__status__ = "Production


## Code References
#
# 1.https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/  
#
# ## end of references


###################################################
# LSTM model without Bayesian optimization
###################################################

#%% We will import required librabries as we progress
import pandas as pd
import numpy as np

# Data - preprocessing section
#%% checking for missing data
df = pd.read_csv("AAPL_2014_2019.csv",header=0, index_col=0, parse_dates=True)
# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']
df.isnull().sum()


#%% split into train validate and test
from math import ceil
data_all_values = df.values
data_split_1 = ceil(len(data_all_values)*0.60)
data_train = data_all_values[:data_split_1, :]
data_temp = data_all_values[data_split_1:, :]
data_split_2 = ceil(len(data_temp)*0.50)
data_valid = data_temp[:data_split_2, :]
data_test = data_temp[data_split_2:, :]
#%% Normalize train
from sklearn.preprocessing import MinMaxScaler
#values=data_train.values
values = data_train.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)

#%% Normalize valid
from sklearn.preprocessing import MinMaxScaler
#values=data_valid.values
values = data_valid.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_valid = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)

#%% Normalize test
from sklearn.preprocessing import MinMaxScaler
#values=data_test.values
values = data_test.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)

# %% define timer series to supervised learning function
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#%% dropping variables
df=None
MinMaxScaler=None
#scaler=None
values=None
data_split_1=None
data_split_2=None
data_temp=None
data_test=None
data_train=None
data_valid=None
DataFrame=None
data_all_values=None
#%% store in a file
import os.path
import csv
fileName='look_back_tuning.csv'
if os.path.exists(fileName):
    os.remove(fileName)
    print('Path exists')
else:
    with open(fileName, 'a') as csvfile:
        filewriter = csv.writer(csvfile)
        row=['look_back', 'rmse','r2','mse','mae','forecast_error']
        filewriter.writerow(row)
    csvfile.close()


#%% define our lagged observations
#It is standard practice in time series forecasting to use 
# lagged observations (e.g. t-1) as input variables to forecast 
# the current time step (t)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import DataFrame
from keras import optimizers
from math import ceil
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

#x=4
look_back=range(4,101)

for x in look_back:
    n_features=6
    data_scaled_train=series_to_supervised(scaled_train,x,1)
    data_scaled_valid=series_to_supervised(scaled_valid,x,1)
    data_scaled_test=series_to_supervised(scaled_test,x,1)
    #print(data.head())


    # %% drop columns we don't want to predict
    data_scaled_train.drop(data_scaled_train.columns[range(len(data_scaled_train.columns)-(n_features-1),len(data_scaled_train.columns),1)], axis=1, inplace=True)
    data_scaled_valid.drop(data_scaled_valid.columns[range(len(data_scaled_valid.columns)-(n_features-1),len(data_scaled_valid.columns),1)], axis=1, inplace=True)
    data_scaled_test.drop(data_scaled_test.columns[range(len(data_scaled_test.columns)-(n_features-1),len(data_scaled_test.columns),1)], axis=1, inplace=True)
    #print(data.head())
    #%%
    # getting values
    data_scaled_train_values=data_scaled_train.values
    data_scaled_valid_values=data_scaled_valid.values
    data_scaled_test_values=data_scaled_test.values
    #%% split into input and outputs
    n_obs = x * n_features
    train_X, train_y = data_scaled_train_values[:, :n_obs], data_scaled_train_values[:, -n_features]
    valid_X, valid_y = data_scaled_valid_values[:, :n_obs], data_scaled_valid_values[:, -n_features]
    test_X, test_y = data_scaled_test_values[:, :n_obs], data_scaled_test_values[:, -n_features]
    #print(train_X.shape, len(train_X), train_y.shape)
    #%% Reshape input
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], x, n_features))
    valid_X = valid_X.reshape((valid_X.shape[0], x, n_features))
    test_X = test_X.reshape((test_X.shape[0], x, n_features))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # %% building model

    # design network
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy','mae'])
    #%% # fit network 
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit(train_X, train_y, epochs=700, batch_size=32,callbacks=callbacks,validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
    #%%
    # make a prediction
    yhat = model.predict(test_X) 
    test_X = test_X.reshape((test_X.shape[0], x*n_features))

    # invert scaling for forecast  
    #inv_yhat = predicted
    inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    #inv_y= actual test
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('look_back {} , the rmse is {}'.format(x,rmse))
    # R2 score
    from sklearn.metrics import r2_score
    r2=r2_score(inv_y, inv_yhat)
    print('r2: %f' % r2)
    #Mean Sqaured Error
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(inv_y, inv_yhat)
    print('MSE: %f' % mse)
    # Mean Absolute Error
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('MAE: %f' % mae)

    #Mean Forecast Error
    forecast_errors = [inv_y[i]-inv_yhat[i] for i in range(len(inv_y))]
    bias = sum(forecast_errors) * 1.0/len(inv_y)
    print('Bias: %f' % bias)

    model=None
    test_X=None
    test_y=None
    inv_yhat=None
    inv_y=None
    yhat=None
    train_X=None
    train_y=None
    valid_X=None
    valid_Y=None
    data_scaled_train=None
    data_scaled_valid=None
    data_all_values=None
    data=None
    data_scaled_train_values=None
    data_scaled_valid_values=None
    data_scaled_test_values=None
    with open(fileName, 'a',newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow([x,rmse,r2,mse,mae,bias])
        writeFile.close()

    rmse=None
    r2=None
    mse=None
    mae=None
    bias=None

# %%
# from matplotlib import pyplot 
# r = range(0,185)
# pyplot.plot(r, inv_y, 'r--', r, inv_yhat, 'b--')

# %%