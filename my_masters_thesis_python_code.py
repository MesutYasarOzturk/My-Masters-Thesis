# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:13:43 2020

@author: Mesut Yasar Ozturk
"""

#==================================
#ARIMA
#==================================

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

hkexdata = pd.read_csv("C:/Users/Yasar M. Ozturk/Desktop/QEA Files/My Thesis/hkex.csv", index_col='Date', parse_dates=['Date'])

pd.set_option('display.max_columns', 100)
pd.set_option('precision', 4)

#Plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close price')
plt.plot(hkexdata['Close'], color='blue')
plt.title('HKEX close price')
plt.show()

df_close = hkexdata['Close']

#Test for staionarity
def stationarity_check(timeseries):
    #Determing rolling statistics
    rollingmean = timeseries.rolling(12).mean()
    rollingstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rollingmean, color='red', label='Rolling Mean')
    plt.plot(rollingstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and the Standard Deviation')
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adftest = adfuller(timeseries,autolag='AIC')
    result = pd.Series(adftest[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adftest[4].items():
        result['critical value (%s)'%key] =  values
    print(result)
    
stationarity_check(df_close)

#Decompose time series
result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(10, 6)

#Take a log of the series, find the rolling mean and std. of the series.
from pylab import rcParams
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()

#Plot the findings
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Log of Close price')
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(moving_avg, color="blue", label = "Mean")
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.legend()
plt.show()

#Split data into train-test sets
strain = '2015-01-01'
etrain = '2018-12-31'
stest = '2019-01-01'      
etest = '2019-12-31'
train_data = df_log.loc[strain:etrain]
test_data = df_log.loc[stest:etest]

print('train_data:', train_data.shape)  
print('test_data:', test_data.shape) 

#Plot split data
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close price')
plt.plot(df_log, 'blue', label='Train data')
plt.plot(test_data, 'red', label='Test data')
plt.legend()

#Deternine optimal values for p,q minimizing AIC; let the algorithm decide d. Use ADF test.
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find  optimal 'd'
                      max_p=10, max_q=10, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True) 
print(model_autoARIMA.summary())

#Review the residual plots
model_autoARIMA.plot_diagnostics(figsize=(10,6))
plt.show()

#Create ARIMA(p,d,q) model
model = ARIMA(train_data, order=(1, 1, 1))  
fitted = model.fit(disp=-1)
print(fitted.summary())

# Forecast the stock prices on the test set keeping 95% confidence level
fc, se, conf = fitted.forecast(246, alpha=0.05)  
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

train_data_=np.exp(train_data).astype(int)
test_data_=np.exp(test_data).astype(int)
fc_series_=np.exp(fc_series).astype(int)

#Plot actual, predicted prices
plt.figure(figsize=(10,6), dpi=100)
plt.grid(True)
plt.plot(train_data_, color = 'blue', label='Training')
plt.plot(test_data_, color = 'red', label='Actual Stock Price')
plt.plot(fc_series_, color = 'black',label='Predicted Stock Price')
plt.title('HKEX Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#Performance scores
mse = mean_squared_error(test_data_, fc_series_)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data_, fc_series_)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data_, fc_series_))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs( test_data_ - fc_series_)/np.abs(test_data_))
print('MAPE: '+str(mape))

#==================================
#FACEBOOK'S PROPHET
#==================================

#Facebook's Prophet
from fbprophet import Prophet

hkexdata = pd.read_csv("C:/Users/Yasar M. Ozturk/Desktop/QEA Files/My Thesis/hkex.csv", parse_dates=['Date'])
hkexdata.reset_index(drop=False, inplace=True)

#Name the input dataframe’s columns as ds and y.
hkexdata.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
hkexdata = hkexdata.drop(['Open', 'Low', 'Volume', 'High', 'Adj Close'], axis=1)

#Split the series into the training and test sets:
train_indices = hkexdata.ds.apply(lambda x: x.year) < 2019

df_train = hkexdata.loc[train_indices].dropna()
df_test = hkexdata.loc[~train_indices].reset_index(drop=True)
print(df_train.head())
print(df_test.head())

#Create the model and fit the data:
prophet_model = Prophet(seasonality_mode='additive', daily_seasonality = True)
prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
prophet_model.fit(df_train)

#Forecast the HKEX prices and plot the results:
df_future = prophet_model.make_future_dataframe(periods=365)
df_prediction = prophet_model.predict(df_future)
print(df_prediction.head(5))

#Plot predictions
prophet_model.plot(df_prediction)

#Inspect the decomposition of the time series:
prophet_model.plot_components(df_prediction)

#Merge the test set with the forecasts
selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']
df_prediction = df_prediction.loc[:, selected_columns].reset_index(drop=True)
#print(df_pred.head(3))
df_test = df_test.merge(df_prediction, on=['ds'], how='left')
#print(df_test.head(3))
df_test.ds = pd.to_datetime(df_test.ds)
df_test.set_index('ds', inplace=True)
#Print the first 3 rows for actual prices, predictions, and their upper and lower bounds
print(df_test.head(3))

#Plot the test values vs. predictions:                                                            
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10,6)
ax = sns.lineplot(data=df_test[['y', 'yhat_lower','yhat_upper','yhat']])
ax.fill_between(df_test.index, df_test.yhat_lower, df_test.yhat_upper, alpha=0.4)
ax.set(title='Actual close price vs. predicted close price', xlabel='Date', ylabel='Close price')

#Performance scores
mse = mean_squared_error(y_true=df_test['y'], y_pred=df_test['yhat'])
print('MSE: '+str(mse))
mae = mean_absolute_error(y_true=df_test['y'], y_pred=df_test['yhat'])
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y_true=df_test['y'], y_pred=df_test['yhat']))
print('RMSE: '+str(rmse))
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true=df_test['y']
y_pred=df_test['yhat']
print('MAPE: ',mean_absolute_percentage_error(y_true, y_pred))

#==================================
#MACHINE LEARNING MODELS
#==================================

hkexdata = pd.read_csv("C:/Users/Yasar M. Ozturk/Desktop/QEA Files/My Thesis/hkex.csv", header=0, index_col=0 , parse_dates=True, squeeze=True)

# Number of nan values in the data set
hkexdata.isnull().sum()

pd.set_option('display.max_columns', 100)
pd.set_option('precision', 4)

print(type(hkexdata))

print(hkexdata.head()) 
print(hkexdata.tail()) 
print(hkexdata.describe()) 

#Plot actual close prices
plt.plot(hkexdata['Close'], color='blue')
plt.grid(True)
plt.title('Actual close price')
plt.xlabel('Date')
plt.ylabel('Close price')
plt.show()

#Generate features
def original_features(df, df_engineered):
    df_engineered['open'] = df['Open']
    df_engineered['open_1'] = df['Open'].shift(1)
    df_engineered['close_1'] = df['Close'].shift(1)
    df_engineered['high_1'] = df['High'].shift(1)
    df_engineered['low_1'] = df['Low'].shift(1)
    df_engineered['volume_1'] = df['Volume'].shift(1)

def avgerage_price(df, df_engineered):
    df_engineered['avgerage_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_engineered['avgerage_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_engineered['avgerage_price_90'] = df['Close'].rolling(63).mean().shift(1)
    df_engineered['avgerage_price_180'] = df['Close'].rolling(126).mean().shift(1)
    df_engineered['avgerage_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_engineered['ratio_avgerage_price_5_30'] = df_engineered['avgerage_price_5'] / df_engineered['avgerage_price_30']
    df_engineered['ratio_avgerage_price_5_90'] = df_engineered['avgerage_price_5'] / df_engineered['avgerage_price_90']
    df_engineered['ratio_avgerage_price_5_180'] = df_engineered['avgerage_price_5'] / df_engineered['avgerage_price_180']
    df_engineered['ratio_avgerage_price_5_365'] = df_engineered['avgerage_price_5'] / df_engineered['avgerage_price_365']
    df_engineered['ratio_avgerage_price_30_90'] = df_engineered['avgerage_price_30'] / df_engineered['avgerage_price_90']
    df_engineered['ratio_avgerage_price_30_180'] = df_engineered['avgerage_price_30'] / df_engineered['avgerage_price_180']
    df_engineered['ratio_avgerage_price_30_365'] = df_engineered['avgerage_price_30'] / df_engineered['avgerage_price_365']
   
def standard_price(df, df_engineered):
    df_engineered['standard_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_engineered['standard_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_engineered['standard_price_90'] = df['Close'].rolling(63).std().shift(1)
    df_engineered['standard_price_180'] = df['Close'].rolling(126).std().shift(1)
    df_engineered['standard_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_engineered['ratio_standard_price_5_30'] = df_engineered['standard_price_5'] / df_engineered['standard_price_30']
    df_engineered['ratio_standard_price_5_90'] = df_engineered['standard_price_5'] / df_engineered['standard_price_90']
    df_engineered['ratio_standard_price_5_180'] = df_engineered['standard_price_5'] / df_engineered['standard_price_180']
    df_engineered['ratio_standard_price_5_365'] = df_engineered['standard_price_5'] / df_engineered['standard_price_365']
    df_engineered['ratio_standard_price_30_90'] = df_engineered['standard_price_30'] / df_engineered['standard_price_90']
    df_engineered['ratio_standard_price_30_180'] = df_engineered['standard_price_30'] / df_engineered['standard_price_180']
    df_engineered['ratio_standard_price_30_365'] = df_engineered['standard_price_30'] / df_engineered['standard_price_365']
        
    
def avgerage_volume(df, df_engineered):
    df_engineered['avgerage_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_engineered['avgerage_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_engineered['avgerage_volume_90'] = df['Volume'].rolling(63).mean().shift(1)
    df_engineered['avgerage_volume_180'] = df['Volume'].rolling(126).mean().shift(1)
    df_engineered['avgerage_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_engineered['ratio_avgerage_volume_5_30'] = df_engineered['avgerage_volume_5'] / df_engineered['avgerage_volume_30']
    df_engineered['ratio_avgerage_volume_5_90'] = df_engineered['avgerage_volume_5'] / df_engineered['avgerage_volume_90']
    df_engineered['ratio_avgerage_volume_5_180'] = df_engineered['avgerage_volume_5'] / df_engineered['avgerage_volume_180']
    df_engineered['ratio_avgerage_volume_5_365'] = df_engineered['avgerage_volume_5'] / df_engineered['avgerage_volume_365']
    df_engineered['ratio_avgerage_volume_30_90'] = df_engineered['avgerage_volume_30'] / df_engineered['avgerage_volume_90']
    df_engineered['ratio_avgerage_volume_30_180'] = df_engineered['avgerage_volume_30'] / df_engineered['avgerage_volume_180']
    df_engineered['ratio_avgerage_volume_30_365'] = df_engineered['avgerage_volume_30'] / df_engineered['avgerage_volume_365']
    
def standard_volume(df, df_engineered):
    df_engineered['standard_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_engineered['standard_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_engineered['standard_volume_90'] = df['Volume'].rolling(63).std().shift(1)
    df_engineered['standard_volume_180'] = df['Volume'].rolling(126).std().shift(1)
    df_engineered['standard_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_engineered['ratio_standard_volume_5_30'] = df_engineered['standard_volume_5'] / df_engineered['standard_volume_30']
    df_engineered['ratio_standard_volume_5_90'] = df_engineered['standard_volume_5'] / df_engineered['standard_volume_90']
    df_engineered['ratio_standard_volume_5_180'] = df_engineered['standard_volume_5'] / df_engineered['standard_volume_180']
    df_engineered['ratio_standard_volume_5_365'] = df_engineered['standard_volume_5'] / df_engineered['standard_volume_365']
    df_engineered['ratio_standard_volume_30_90'] = df_engineered['standard_volume_30'] / df_engineered['standard_volume_90']
    df_engineered['ratio_standard_volume_30_180'] = df_engineered['standard_volume_30'] / df_engineered['standard_volume_180']
    df_engineered['ratio_standard_volume_30_365'] = df_engineered['standard_volume_30'] / df_engineered['standard_volume_365']
    
def return_feature(df, df_engineered):
    df_engineered['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_engineered['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_engineered['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_engineered['return_90'] = ((df['Close'] - df['Close'].shift(63)) / df['Close'].shift(63)).shift(1)
    df_engineered['return_180'] = ((df['Close'] - df['Close'].shift(126)) / df['Close'].shift(126)).shift(1)
    df_engineered['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_engineered['mov_avg_5'] = df_engineered['return_1'].rolling(5).mean().shift(1)
    df_engineered['mov_avg_30'] = df_engineered['return_1'].rolling(21).mean().shift(1)
    df_engineered['mov_avg_90'] = df_engineered['return_1'].rolling(63).mean().shift(1)
    df_engineered['mov_avg_180'] = df_engineered['return_1'].rolling(126).mean().shift(1)
    df_engineered['mov_avg_365'] = df_engineered['return_1'].rolling(252).mean().shift(1)
    
def generate_features(df):
    df_engineered = pd.DataFrame()
    #6 original features
    original_features(df, df_engineered)
    #add 59 generated features
    avgerage_price(df, df_engineered)
    avgerage_volume(df, df_engineered)
    standard_price(df, df_engineered)
    standard_volume(df, df_engineered)
    return_feature(df, df_engineered)
    #extract the target
    df_engineered['close'] = df['Close']
    df_engineered = df_engineered.dropna(axis=0)
    return df_engineered

#Apply feature engineering to the HKEX data 
data = generate_features(hkexdata)
pd.set_option('precision', 4)
print(data.round(decimals=4).head(5))

#Number of nan values in the data set with new generated features
data.isnull().sum()

#Train-test split
strain = '2015-01-01'
etrain = '2018-12-31'
stest = '2019-01-01'      
etest = '2019-12-31'
train_data = data.loc[strain:etrain]

#Drop close column as it is the target, retrieve the rest of the data 
X_train = train_data.drop('close', axis=1).values
y_train = train_data['close'].values
print('X_train data:', X_train.shape)
print('y_train data:', y_train.shape)
  
test_data = data.loc[stest:etest]
X_test = test_data.drop('close', axis=1).values
y_test = test_data['close'].values
print('X_test data:', X_test.shape)  
print('y_test data:', y_test.shape)  

# count the number of nan values in array
print('Number of NaN values in X_train data:', np.count_nonzero(np.isnan(X_train)))
print('Number of NaN values in y_train data:',np.count_nonzero(np.isnan(y_train)))
print('Number of NaN values in X_test data:',np.count_nonzero(np.isnan(X_test)))
print('Number of NaN values in y_test data:',np.count_nonzero(np.isnan(y_test)))

#Implement SGD-based linear Regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
#Scale the data
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_scaled_train = std_scaler.fit_transform(X_train)
X_scaled_test = std_scaler.transform(X_test)
print('Number of NaN values in X_scaled_train data:', np.count_nonzero(np.isnan(X_scaled_train)))
print('Number of NaN values in X_scaled_test data:',np.count_nonzero(np.isnan(X_scaled_test)))

#Set initial hyperparameter values
grid_params = {"alpha": [1e-5, 3e-5, 1e-4, 3e-5, 1e-3],"eta0": [0.01, 0.03, 0.1, 0.001, 0.3],}
lr = SGDRegressor(penalty='l2', max_iter=10000)

#Determine optimal hyperparameter values
grid_searching = GridSearchCV(lr, grid_params, cv=10, scoring='r2')
grid_searching.fit(X_scaled_train, y_train)
print(grid_searching.best_params_)

#Make predictions
best_lr = grid_searching.best_estimator_
lr_predictions = best_lr.predict(X_scaled_test)

#Plot the predictions
plt.plot(lr_predictions, color='red')
plt.grid(True)
plt.ylabel('Close price')
plt.xlabel('Days from 2019 to 2020')
plt.title('SGD-based linear regression predictions')
plt.show()

plt.plot(y_test, color='blue')
plt.grid(True)
plt.ylabel('Close price')
plt.xlabel('Days from 2019 to 2020')
plt.title('Actual close price')
plt.show()

#Plot pairwise comparison of RF predictions and the actual values
plt.plot(lr_predictions, 'r--', label='LR', color='red')
plt.plot(y_test, label='Actual', color='blue')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('SGD-based linear regression predictions vs. actual close prices')
plt.legend(loc='best')
plt.show()

#Performance scores
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, lr_predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, lr_predictions)))
print('RMSE: {0:.3f}'.format(math.sqrt(mean_squared_error(y_test, lr_predictions))))
mape = np.mean(np.abs( y_test - lr_predictions)/np.abs(y_test))
print('MAPE: '+str(mape))

#Work with Support Vector Regression 
from sklearn.svm import SVR
#Set initial hyperparameter values
grid_params = [{'kernel': ['linear'], 'C': [100, 300, 500],'epsilon': [0.00003, 0.0001, 0.0003, 0.001]},{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-2, 1e-5],'C': [10, 100, 1000,10000], 'epsilon': [0.00003, 0.0001, 0.0003, 0.001]}]

#Determine optimal parameters making use of grid search algorithm
svr = SVR()
grid_searching = GridSearchCV(svr, grid_params, cv=5, scoring='r2')
grid_searching.fit(X_scaled_train, y_train)

#Identify the best SVR model and make predictions of the test samples
print(grid_searching.best_params_)
best_svr = grid_searching.best_estimator_
svr_predictions = best_svr.predict(X_scaled_test)

#Plot the predictions
plt.plot(svr_predictions, color='red')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Support vector regression predictions')
plt.show()

#Plot pairwise comparison of RF predictions and the actual values
plt.plot(svr_predictions, 'r--', label='SVR', color='red')
plt.plot(y_test, label='Actual', color='blue')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Support vector regression predictions vs. actual close prices')
plt.legend(loc='best')
plt.show()

#Performance scores
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, svr_predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, svr_predictions)))
print('RMSE: {0:.3f}'.format(math.sqrt(mean_squared_error(y_test, svr_predictions))))
mape = np.mean(np.abs( y_test - svr_predictions)/np.abs(y_test))
print('MAPE: '+str(mape))

#Forecasting with Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
#Set initial hyperparameter values
grid_params = {'max_depth': [50, 70, 80, 90],'min_samples_split': [2, 5, 10],'max_features': ['auto', 'sqrt'],'min_samples_leaf': [1, 3, 5]}

#Determine optimal parameters making use of grid search algorithm
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
grid_searching = GridSearchCV(rf, grid_params, cv=5, scoring='r2', n_jobs=-1)

grid_searching.fit(X_train, y_train)

#Identify the best RF model and make predictions of the test samples
print(grid_searching.best_params_)
best_rf = grid_searching.best_estimator_
rf_predictions = best_rf.predict(X_test)

#Plot the predictions
plt.plot(rf_predictions, color='red')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Random forest regression predictions')
plt.show()

#Plot pairwise comparison of RF predictions and the actual values
plt.plot(rf_predictions, 'r--', label='RF regression', color='red')
plt.plot(y_test, label='Actual', color='blue')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Random forest predictions vs. actual close prices')
plt.legend(loc='best')
plt.show()

#Performance scores
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, rf_predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, rf_predictions)))
print('RMSE: {0:.3f}'.format(math.sqrt(mean_squared_error(y_test, rf_predictions))))
mape = np.mean(np.abs( y_test - rf_predictions)/np.abs(y_test))
print('MAPE: '+str(mape))

#Multilayer Perceptron Regression
from sklearn.neural_network import MLPRegressor
#Get column means
ta = np.array(X_scaled_train).T.tolist() 
col_means = list(map(lambda x: np.nanmean(x), ta))  

#Replace NaN with column means 
nrows = len(X_scaled_train); ncols = len(X_scaled_train[0]) 
for r in range(nrows):
    for c in range(ncols):
        if np.isnan(X_scaled_train[r][c]):
            X_scaled_train[r][c] = col_means[c]

#Get column means
ta = np.array(y_train).T.tolist()                         
col_means = list(map(lambda x: np.nanmean(x), ta))  
print("column means:", col_means)

#Replace NaN with column means  
nrows = len(y_train); ncols = y_train[0].size 
for r in range(nrows):
    if np.isnan(y_train[r]):
        y_train[r] = col_means[r]
        
#Get column means
ta = np.array(X_scaled_test).T.tolist()                          
col_means = list(map(lambda x: np.nanmean(x), ta))  

#Replace NaN with column means  
nrows = len(X_scaled_test); ncols = len(X_scaled_test[0]) ; 
for r in range(nrows):
    for c in range(ncols):
        if np.isnan(X_scaled_test[r][c]):
            X_scaled_test[r][c] = col_means[c]

#Set initial hyperparameter values
grid_params = {'hidden_layer_sizes': [(90, 10), (10, 10)],'activation': ['logistic', 'tanh', 'relu'],'solver': ['sgd', 'adam'],'learning_rate_init': [0.0001, 0.0003],'alpha': [0.0001],'batch_size': [10, 30]}

#Determine optimal parameters making use of grid search algorithm

nn = MLPRegressor(random_state=12, max_iter=100)
grid_searching = GridSearchCV(nn, grid_params, cv=5, scoring='r2', n_jobs=-1)

grid_searching.fit(X_scaled_train, y_train)

#Identify the best RF model and make predictions of the test samples
print(grid_searching.best_params_)
best_nn = grid_searching.best_estimator_
nn_predictions = best_nn.predict(X_scaled_test)

#Plot the predictions
plt.plot(nn_predictions, color='red')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Neural network- MPL regression predictions')
plt.show()

#Plot pairwise comparison of NN predictions and the actual values
plt.plot(nn_predictions, 'r--', label='MLP regression', color='red')
plt.plot(y_test, label='Actual', color='blue')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.title('Neural network predictions vs. actual close prices')
plt.legend(loc='best')
plt.show()

print('MSE: {0:.3f}'.format(mean_squared_error(y_test, nn_predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, nn_predictions)))
print('RMSE: {0:.3f}'.format(math.sqrt(mean_squared_error(y_test, nn_predictions))))
mape = np.mean(np.abs( y_test - nn_predictions)/np.abs(y_test))
print('MAPE: '+str(mape))

#Plot all ML algorithms' predictions
plt.title('The Hong Kong Exchanges and Clearing Limited close price prediction vs Truth')
plt.plot(y_test, label='Actual')
plt.grid(True)
plt.plot(lr_predictions, 'r--', label='Linear regression', color='green')
plt.plot(rf_predictions, 'r--', label='Random forest', color='red')
plt.plot(svr_predictions, 'r--', label='Support vector regression', color='black')
plt.plot(nn_predictions, 'r--', label='Neural network', color='yellow')
plt.grid(True)
plt.xlabel('Days from 2019 to 2020')
plt.ylabel('Close price')
plt.legend(loc='best')





   



















