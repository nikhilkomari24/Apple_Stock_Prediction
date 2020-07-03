"""arima_lstm_bayesian.py: Predicting stock prices using ARIMA+ LSTM and Bayesian optimizer"""
__authors__ = "Keith Herbert, Rasphinder Jagait, Nikhil Komari, Sai Bhasker Raju"
__copyright__ = "Copyright 2019, Comparative Analysis of Currency Prediction Methods"
__license__ = "GPL"
__version__ = "1.0.0"
__maintained_by__ = "Keith Herbert, Rasphinder Jagait, Nikhil Komari, Sai Bhasker Raju"
__email__ = "kherbe@uwo.ca, rjagait@uwo.ca, nkomari@uwo.ca, schittap@uwo.ca"
__status__ = "Production


## Code References
#
# 1.https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/  
# 2.https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_08_4_bayesian_hyperparameter_opt.ipynb 
# 3.https://github.com/imhgchoi/ARIMA-LSTM-hybrid-corrcoef-predict
#
# ## end of references

###################################################
# Decomposing the signal to train and test on individual models
###################################################

# Decomposing individually for visual representation
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("AAPL_974.csv",header=0, index_col=0, parse_dates=True)

print("Decomposing Volume signal considering frequency of a week");
#%% visualize
df.head()
df_decomp = seasonal_decompose(df['Volume'],freq=5)
df_decomp.plot()
# df_resid['Volume'].plot()
pyplot.show()


# Decomposing entire dataset and write in csv files
df_decomp = seasonal_decompose(df,freq=5)
df_trend = df_decomp.trend; #ARIMA
df_seasonal = df_decomp.seasonal; #ARIMA
df_resid = df_decomp.resid; #LSTM

df_arima = df_trend.add(df_seasonal);
df_arima = df_arima.fillna(0);
df_lstm = df_resid;
df_lstm = df_lstm.fillna(0);

df_arima.to_csv('df_arima.csv', sep=',')
df_lstm.to_csv('df_lstm.csv', sep=',')


###################################################
# LSTM model with BayesianOptimization on the trend of the signal
###################################################

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from matplotlib import pyplot

look_back=49
n_features=6
scaler = MinMaxScaler(feature_range=(0, 1))

class Data_Eval:
  def __init__(self):
    self.rmse = None
    self.data_inv_y = None
    self.data_inv_yhat = None
    self.history= None

data_eval_best = Data_Eval()
data_eval_worst = Data_Eval()

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

def get_input_datasets(use_bfloat16=False):
    # """Downloads the MNIST dataset and creates train and eval dataset objects.

    # Args:
    #   use_bfloat16: Boolean to determine if input should be cast to bfloat16

    # Returns:
    #   Train dataset, eval dataset and input shape.

    # Data - preprocessing section
    #% checking for missing data

    df = pd.read_csv("df_LSTM.csv",header=0, index_col=0, parse_dates=True)
    df.isnull().sum()
    df.head()
    values = df.values

    #% Normalize
    # ensure all data is float

    values=df.values
    values = values.astype('float32')

    # normalize features
    scaled = scaler.fit_transform(values)

    #% define our lagged observations
    #It is standard practice in time series forecasting to use 
    # lagged observations (e.g. t-1) as input variables to forecast 
    # the current time step (t)
    data=series_to_supervised(scaled,look_back, 1)
    print(data.head())

    # % drop columns we don't want to predict
    # data.drop(data.columns[range(len(data.columns)-5,len(data.columns),1)], axis=1, inplace=True)
    # print(data.head())

    from math import ceil
    data_all_values = data.values
    # Ratio - https://www.coursera.org/lecture/deep-neural-network/train-dev-test-sets-cxG1s
    data_split_1 = ceil(len(data_all_values)*0.60)
    data_train = data_all_values[:data_split_1, :]
    data_temp = data_all_values[data_split_1:, :]

    data_split_2 = ceil(len(data_temp)*0.50)
    data_valid = data_temp[:data_split_2, :]
    data_test = data_temp[data_split_2:, :]

    #% split into input and outputs
    n_features=6
    n_obs = look_back * n_features
    train_X, train_y = data_train[:, :n_obs], data_train[:, -n_features]
    valid_X, valid_y = data_valid[:, :n_obs], data_valid[:, -n_features] 
    test_X, test_y = data_test[:, :n_obs], data_test[:, -n_features]
    # print(train_X.shape, len(train_X), train_y.shape)

    #% Reshape input
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], look_back, n_features))
    valid_X = valid_X.reshape((valid_X.shape[0], look_back, n_features))
    test_X = test_X.reshape((test_X.shape[0], look_back, n_features))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def get_model(train_X, train_y, valid_X, valid_y, test_X, test_y, dense_1_neurons=128, batch=16):
    """Builds a Sequential model

    Args:

    Returns:
      a Keras model

    """
    model = Sequential()
    model.add(LSTM(dense_1_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

train_X, train_y, valid_X, valid_y, test_X, test_y = get_input_datasets()

def fit_with(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose, dense_1_neurons_x128, batch, ep):

    dense_1_neurons = max(int(dense_1_neurons_x128 * 128), 128)
    model = get_model(train_X, train_y, valid_X, valid_y, test_X, test_y, dense_1_neurons, batch)

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    #history = model.fit(train_X, train_y, epochs=int(ep), batch_size=int(batch),callbacks=callbacks,validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
    history = model.fit(train_X, train_y, epochs=int(ep), batch_size=int(batch), validation_data=(valid_X, valid_y), verbose=2, shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], look_back*n_features))
    #%
    # invert scaling for forecast
    from numpy import concatenate
    inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    #%
    # calculate RMSE
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    if data_eval_worst.rmse == None or data_eval_worst.rmse>rmse :
        data_eval_worst.rmse = rmse
        data_eval_worst.history = history
        data_eval_worst.data_inv_y = inv_y
        data_eval_worst.data_inv_yhat = inv_yhat

    if data_eval_best.rmse == None or data_eval_best.rmse<rmse :
        data_eval_best.rmse = rmse
        data_eval_best.history = history
        data_eval_best.data_inv_y = inv_y
        data_eval_best.data_inv_yhat = inv_yhat
    
    return (rmse*-1)

def plot_data(data_all):

    # Plot data
    # Plot each column
    i = 1
    pyplot.figure()
    for col in len(data.columns):
        pyplot.subplot(len(column_plot), 1, i)
        pyplot.plot(data_all_values[:, col])
        pyplot.title(data_all.columns[col], y=0.5, loc='right')
        i += 1
    pyplot.show()

    # Box plot
    data_all.plot(kind='box', subplots=True, layout=(4,2), sharex=False, sharey=False)
    pyplot.show()

    # Histogram plot
    data_all.hist(layout=(4,2))
    pyplot.show()

    scatter_matrix(data_all)
    pyplot.show()

def plot_results(data_eval):

    rmse = data_eval.rmse
    data_inv_y = data_eval.data_inv_y
    data_inv_yhat = data_eval.data_inv_yhat
    data_fit_history = data_eval.history

    print('RMSE: %.3f' % rmse)

    # Plot history
    pyplot.title('RMSE: %.3f' % rmse)
    pyplot.plot(data_fit_history.history['loss'], label='train')
    pyplot.plot(data_fit_history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    pyplot.title('RMSE: %.3f' % rmse)
    pyplot.plot(data_inv_y, label='Original')
    pyplot.plot(data_inv_yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()

from functools import partial
verbose = 1
fit_with_partial = partial(fit_with, train_X, train_y, valid_X, valid_y, test_X, test_y, verbose)
fit_with_partial(dense_1_neurons_x128=1, batch=16, ep=50)

# The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is `maximize`, which does exactly what you think it does.
# 
# There are many parameters you can pass to maximize, nonetheless, the most important ones are:
# - `n_iter`: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
# - `init_points`: How many steps of **random** exploration you want to perform. Random exploration can help by diversifying the exploration space.

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {
    "dense_1_neurons_x128": (0.9, 3.1),
    "batch": (16, 320),
    "ep": (1,100)
}

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

#optimizer.maximize(init_points=10, n_iter=10,)
optimizer.maximize(init_points=3, n_iter=3,)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)

plot_results (data_eval_worst)
plot_results (data_eval_best)


# Store the lstm prediction
Original_lstm = data_eval_worst.data_inv_y;
Predicted_lstm = data_eval_worst.data_inv_yhat;


# calculate RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(Original_lstm, Predicted_lstm))
print('Test RMSE: %.3f' % rmse)


###################################################
# Rolling Forecast ARIMA model on the seasonality and residual of the signal
###################################################

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read from csv file
df_arima = pd.read_csv("df_arima.csv",header=0, index_col=0, parse_dates=True)
df_arima = df_arima.fillna(0);

# drop columns we don't want to predict
df_arima.drop(df_arima.columns[range(len(df_arima.columns)-5,len(df_arima.columns),1)], axis=1, inplace=True)

# split data for ARIMA
data_all_values_arima = df_arima.values
data_train_split_arima = ceil(len(data_all_values_arima)*0.80)
data_train_arima = data_all_values_arima[:data_train_split_arima, :]
data_test_arima = data_all_values_arima[data_train_split_arima:, :]

history_arima = [x for x in data_train_arima]
predictions_arima = list()
for t in range(len(data_test_arima)):
	model_arima = ARIMA(history_arima, order=(5,1,0)) #(5,1,0))
	model_fit_arima = model_arima.fit(disp=0)
	output_arima = model_fit_arima.forecast()
	yhat_arima = output_arima[0]
	predictions_arima.append(yhat_arima)
	obs_arima = data_test_arima[t]
	history_arima.append(obs_arima)

# Plot graph
data_test_arima = data_test_arima[:len(data_test_arima)-2];
predictions_arima = predictions_arima[:len(predictions_arima)-2]

# calculate MAE & RMSE
error_arima = mean_squared_error(data_test_arima, predictions_arima)
print('Test MSE: %.3f' % error_arima)

rmse_arima = sqrt(mean_squared_error(data_test_arima, predictions_arima))
print('Test RMSE: %.3f' % rmse_arima)

# Plot graph
x_data_arima = range(len(data_all_values_arima)-2)
plt.plot(data_train_arima, label="Train Set")
plt.plot(x_data_arima[data_train_split_arima:], data_test_arima, label="Test Set")
plt.plot(x_data_arima[data_train_split_arima:], predictions_arima, label="Prediction")
plt.title("Open") 
plt.xlabel("Date") 
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()


###################################################
# Plotting graphs and accuracy measures for individual models 
# and for the final combined prediction
###################################################

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ARIMA

Original_arima = data_test_arima;
Predicted_arima = predictions_arima;
print('ARIMA RMSE: %.3f' % sqrt(mean_squared_error(data_test_arima, predictions_arima)))
print('ARIMA MAE: %.3f' % mean_absolute_error(data_test_arima, predictions_arima))
print('ARIMA rsquare: %.3f' % r2_score(data_test_arima, predictions_arima))

plt.plot(data_test_arima, label="Original")
plt.plot(predictions_arima, label="Predicted")
plt.title("ARIMA Output") 
plt.xlabel("Date") 
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()

# LSTM

Original_lstm = data_eval_worst.data_inv_y;
Predicted_lstm = data_eval_worst.data_inv_yhat;
print('LSTM RMSE: %.3f' % sqrt(mean_squared_error(Original_lstm, Predicted_lstm)))
print('LSTM MAE: %.3f' % mean_absolute_error(Original_lstm, Predicted_lstm))
print('LSTM rsquare: %.3f' % r2_score(Original_lstm, Predicted_lstm))

pyplot.plot(Original_lstm, label='Original')
pyplot.plot(Predicted_lstm, label='Predicted')
pyplot.title("LSTM Output") 
pyplot.xlabel("Date") 
pyplot.ylabel("Price")
pyplot.legend(loc="upper left")
pyplot.show()

# Combined
Original_arima_array = np.array(Original_arima);
Predicted_arima_array = np.array(Predicted_arima);
Original_arima1= Original_arima_array[:(Original_arima_array.size)-7].transpose().flatten();
Predicted_arima1= Predicted_arima_array[:(Predicted_arima_array.size)-7].transpose().flatten();

Original_combined = np.add(np.array(Original_lstm),Original_arima1);
Predicted_combined = np.add(np.array(Predicted_lstm),Predicted_arima1);
print('Combined RMSE: %.3f' % sqrt(mean_squared_error(Original_combined, Predicted_combined)))
print('Combined MAE: %.3f' % mean_absolute_error(Original_combined, Predicted_combined))
print('Combined rsquare: %.3f' % r2_score(Original_combined, Predicted_combined))

pyplot.plot(Original_combined, label='Original')
pyplot.plot(Predicted_combined, label='Predicted')
pyplot.title("Combined Output") 
pyplot.xlabel("Date") 
pyplot.ylabel("Price")
pyplot.legend(loc="upper left")
pyplot.show()