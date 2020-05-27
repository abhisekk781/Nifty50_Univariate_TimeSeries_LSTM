from pandas import DataFrame
from pandas import Series
from pandas import concat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
 
def series_to_seq(data, n_in=1, n_out=1, dropnan=True):
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


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def prepare_data(series, n_lag=1, n_seq=1):
# extract raw values
    raw_values = series.values
   # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
# rescale values to -1, 1
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
   # transform into supervised learning problem X, y
    supervised = series_to_seq(scaled_values, n_lag, n_seq)
    return supervised



# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		scaler = MinMaxScaler()
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted


#Rebuild Differencing
def rebuild_difference(forecast, first_element_original):
    forecast=np.array(forecast)
    cumul = forecast.cumsum()
    cumul=Series(cumul)
    x=cumul.fillna(0) + first_element_original
    x=list(x)
    x.insert(0,first_element_original)
    return Series(x)

#Logarithmic Differencing
def log_difference(series):
    log=np.log(series)
    log_diff=difference(log)
    return Series(log_diff)

#Rebuild Logarithmic Differencing
def rebuild_log_diffed(series,first_element_original):
    reb=rebuild_difference(series,np.log(first_element_original))
    rebbed=[np.round(np.exp(x)) for x in reb]
    return Series(rebbed)

#Add mean to the series
def add_mean(series):
    mean=np.mean(series)
    series=list(series)
    series.append(mean)
    return Series(series)

#remove the last element of series
def remove_last(series):
    series=list(series)
    series=series[:-1]
    return Series(series)

#Seasonality removal and rebuilder

#Generate curve to fit the data
def generate_curve(series,degree):
    X = [i%365 for i in range(0, len(series.values))] # Remove yearly trends
    y = series.values
    coef = np.polyfit(X, y, int(degree))
    #Create curve
    curve = []
    constant_term =coef[-1][0] # This is the constant term
    degree=4
    for i in range(len(X)):
        product=constant 
        for j in range(degree):
            product += np.multiply(np.power(X[i],degree-j),coef[j][0])
        curve.append(product)    
    return curve


#Remove seasonality
def seasonality_difference(series,curve):
    y=series.values
    diff = []
    for i in range(len(y)):
        value = y[i] - curve[i]
        diff.append(value)
    return diff


#rebuild seasonality
def rebuild_seasonality(forecast,curve):
    forecast=list(forecast)
    undiff=[]
    for i in range(len(forecast)):
        value=forecast[i]+curve[i]
        undiff.append(value)
    return undiff


#Remove Outliers using LOF
def remove_outliers_LOF(X):
    X=X.reshape(-1,1)
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    mask = yhat != -1
    return X[mask]


#Remove Outliers using quartile range
def remove_outliers_IQR(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

#create encoding of string objects
def encode(series):
    encoder = LabelEncoder()
    return encoder.fit_transform(series)


#One hot encoding
def one_hot_encode(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I