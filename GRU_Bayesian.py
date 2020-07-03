"""gru_bayesian.py: Predicting stock prices using GRU and Bayesian optimizer"""
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
# 2.https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_08_4_bayesian_hyperparameter_opt.ipynb 
#
# ## end of references


###################################################
# GRU model with Bayesian optimization
###################################################

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import ceil
from numpy import concatenate

look_back=49
n_features=6
scaler = MinMaxScaler(feature_range=(0, 1))
surpress_plots = False

class Data_Eval:
  def __init__(self):
    self.mae = None
    self.rmse = None
    self.mfe = None
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

def plot_data(df):
    values=df.values
    # plot each column
    pyplot.figure()
    for i in range(0, len(df.columns)):
        pyplot.subplot(len(df.columns), 1, (i+1))
        pyplot.plot(values[:, i])
        pyplot.title(df.columns[i], y=0.5, loc='right')

    pyplot.savefig("data_groups.png")
    if surpress_plots == False:
        pyplot.show()

    # Box plot
    df.plot(kind='box', subplots=True, layout=(4,2), sharex=False, sharey=False)
    pyplot.savefig("data_box.png")
    if surpress_plots == False:
        pyplot.show()

    # Histogram plot
    df.hist(layout=(4,2))
    pyplot.savefig("data_histogram.png")
    if surpress_plots == False:
        pyplot.show()

    scatter_matrix(df)
    pyplot.savefig("data_scatter.png")
    if surpress_plots == False:
        pyplot.show()

def get_input_datasets(use_bfloat16=False):
    # """Downloads the MNIST dataset and creates train and eval dataset objects.

    # Args:
    #   use_bfloat16: Boolean to determine if input should be cast to bfloat16

    # Returns:
    #   Train dataset, eval dataset and input shape.

    # Data - preprocessing section
    #% checking for missing data


    #% checking for missing data
    df = pd.read_csv("data.csv",header=0, index_col=0, parse_dates=True)
    df.isnull().sum()

    #%% split into train validate and test
    data_all_values = df.values
    data_split_1 = ceil(len(data_all_values)*0.60)
    data_train = data_all_values[:data_split_1, :]
    data_temp = data_all_values[data_split_1:, :]
    data_split_2 = ceil(len(data_temp)*0.50)
    data_valid = data_temp[:data_split_2, :]
    data_test = data_temp[data_split_2:, :]

    values = data_train.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)
    values = data_valid.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_valid = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)
    values = data_test.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = pd.DataFrame(scaler.fit_transform(values), columns = df.columns)

    data_scaled_train=series_to_supervised(scaled_train,look_back,1)
    data_scaled_valid=series_to_supervised(scaled_valid,look_back,1)
    data_scaled_test=series_to_supervised(scaled_test,look_back,1)

    # drop columns we don't want to predict
    data_scaled_train.drop(data_scaled_train.columns[range(len(data_scaled_train.columns)-(n_features-1),len(data_scaled_train.columns),1)], axis=1, inplace=True)
    data_scaled_valid.drop(data_scaled_valid.columns[range(len(data_scaled_valid.columns)-(n_features-1),len(data_scaled_valid.columns),1)], axis=1, inplace=True)
    data_scaled_test.drop(data_scaled_test.columns[range(len(data_scaled_test.columns)-(n_features-1),len(data_scaled_test.columns),1)], axis=1, inplace=True)

    # getting values
    data_scaled_train_values=data_scaled_train.values
    data_scaled_valid_values=data_scaled_valid.values
    data_scaled_test_values=data_scaled_test.values

    n_obs = look_back * n_features
    train_X, train_y = data_scaled_train_values[:, :n_obs], data_scaled_train_values[:, -n_features]
    valid_X, valid_y = data_scaled_valid_values[:, :n_obs], data_scaled_valid_values[:, -n_features]
    test_X, test_y = data_scaled_test_values[:, :n_obs], data_scaled_test_values[:, -n_features]
    #print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], look_back, n_features))
    valid_X = valid_X.reshape((valid_X.shape[0], look_back, n_features))
    test_X = test_X.reshape((test_X.shape[0], look_back, n_features))

    plot_data(df)
    return train_X, train_y, valid_X, valid_y, test_X, test_y, scaler

def get_model(train_X, train_y, valid_X, valid_y, test_X, test_y, dense_1_neurons=128, batch=16):
    """Builds a Sequential model

    Args:

    Returns:
      a Keras model

    """
    model = Sequential()
    model.add(GRU(dense_1_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

train_X, train_y, valid_X, valid_y, test_X, test_y, scaler = get_input_datasets()

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

    # calculate Errors
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    mae = mean_squared_error(inv_y, inv_yhat)
    forecast_errors = [inv_y[i]-inv_yhat[i] for i in range(len(inv_y))]
    mfe = sum(forecast_errors) * 1.0/len(inv_y)

    print('Test RMSE: %.3f' % rmse)
    print('Test R^2: %.3f' % mae) 
    print('MFE: %.3f' % mfe)

    if data_eval_worst.rmse == None or data_eval_worst.rmse>rmse :
        data_eval_worst.mfe = mfe
        data_eval_worst.mae = mae
        data_eval_worst.rmse = rmse
        data_eval_worst.history = history
        data_eval_worst.data_inv_y = inv_y
        data_eval_worst.data_inv_yhat = inv_yhat

    if data_eval_best.rmse == None or data_eval_best.rmse<rmse :
        data_eval_best.mfe = mfe
        data_eval_best.mae = mae
        data_eval_best.rmse = rmse
        data_eval_best.history = history
        data_eval_best.data_inv_y = inv_y
        data_eval_best.data_inv_yhat = inv_yhat
    
    return (rmse*-1)

def plot_results(data_eval, pre):

    rmse = data_eval.rmse
    mae = data_eval.mae
    mfe = data_eval.mfe
    data_inv_y = data_eval.data_inv_y
    data_inv_yhat = data_eval.data_inv_yhat
    data_fit_history = data_eval.history

    print('RMSE: %.3f' % rmse)

    # Plot history
    np.savetxt(pre+"_history_inv_y.csv", data_inv_y, delimiter=",")
    np.savetxt(pre+"_history_inv_yhat.csv", data_inv_yhat, delimiter=",")

    pyplot.title("Learning Rate - RMSE: {:.3f}; MAE: {:.3f}; MFE: {:.3f}".format(rmse, mae, mfe))
    pyplot.xlabel("Epoch") 
    pyplot.ylabel("Loss")
    pyplot.plot(data_fit_history.history['loss'], label='Train')
    pyplot.plot(data_fit_history.history['val_loss'], label='Validation')
    pyplot.legend()
    pyplot.savefig(pre+"_lr.png")
    if surpress_plots == False:
        pyplot.show()

    pyplot.title("Apple Stock - RMSE: {:.3f}; MAE: {:.3f}; MFE: {:.3f}".format(rmse, mae, mfe))
    pyplot.plot(data_inv_y, label='Original')
    pyplot.plot(data_inv_yhat, label='Predicted')
    pyplot.xlabel("Time") 
    pyplot.ylabel("Price")
    pyplot.savefig(pre+"_predict.png")

    pyplot.legend()
    if surpress_plots == False:
        pyplot.show()

from functools import partial
verbose = 1
fit_with_partial = partial(fit_with, train_X, train_y, valid_X, valid_y, test_X, test_y, verbose)
#fit_with_partial(dense_1_neurons_x128=1, batch=16, ep=50)

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

optimizer.maximize(init_points=3, n_iter=3,)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)
plot_results (data_eval_worst, "best")
plot_results (data_eval_best, "worst")