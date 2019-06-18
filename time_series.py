from __future__ import unicode_literals, division, print_function

#For handling numbers
import numpy as np

#For accepting command line arguments
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Import main module
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import data_handler
import config

path_to_model = './arima/saved_lag_24'

#Importing a few hyperparameters
seq_length = config.seq_length
output_seq_length = config.output_seq_length
order = config.order

#Workaround around ARIMA's loading bug
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__

#Fetch and split data
raw_x, raw_y = data_handler.fetch_data()
train_x, test_x = data_handler.split_data(raw_x)
train_y, test_y = data_handler.split_data(raw_y)

#Difference the whole dataset to remove seasonality
history = np.append(train_x, test_x)
differenced = data_handler.remove_seasonality(history, lag=order[0]).values
train = sys.argv[1]

if (train == 'train'):
    #Declare model
    model = ARIMA(differenced[:len(train_x)], order=order)
    #Fit model and save parameters
    model = model.fit(disp=1)
    model.save(path_to_model)
elif (train == 'mape'):
    mape_sum = 0
    mape_scores = list()
    #Iterates through the test set, predicting and calculating MAPE
    for seq in range(len(test_y)//output_seq_length):
        #Start with the first test week
        #We append the training set since our values will depend on those values
        #After each sequence is evaluated, its actual values become part of the training data
        model = ARIMA(differenced[:len(train_x) + (seq+1) * output_seq_length], order=order)
        #Fit model to previous values
        model = model.fit()
        preds = model.forecast(output_seq_length)[0]
        #We convert the np.float64 to integers due to the weird behaviour of matplotlib
        #with float64 types
        pred = list()
        for i in range(len(preds)):
            pred.append(int(preds[i]))

        #Add the actual values for history, as for the training data
        history_ = history[:len(train_x) + (seq+1) * output_seq_length]
        #Add seasonality, using the history
        for prediction in pred:
            np.append(history_, data_handler.add_seasonality(history_,prediction,order[0]))
        #Take the relevant values
        pred = np.array(history_[-output_seq_length:])
        #Retrieve the actual target values
        test_y_ = test_y[seq * output_seq_length:]
        #Calculate MAPE and store in list
        mape_score = data_handler.MAPE(test_y_.reshape(-1)[:output_seq_length],pred)
        mape_scores.append(mape_score)
        mape_sum += mape_score
    #Print a list of MAPE scores for the test data ad their average
    print(round(mape_sum/(len(test_y)//output_seq_length), 3))
    print(mape_scores)
elif (train == 'test'):
    #Declare and fit model
    model = ARIMA(differenced, order=order)
    model = ARIMAResults.load(path_to_model)

    #Testing the first sequence
    history = np.append(train_x, test_x[:output_seq_length])
    preds = model.forecast(output_seq_length)[0]
    pred = list()
    
    #Removing float64 types for convenient plotting
    for i in range(len(preds)):
        pred.append(int(preds[i]))

    #Restore seasonality
    for prediction in pred:
        np.append(history, data_handler.add_seasonality(history,prediction,output_seq_length))
    pred = np.array(history[-output_seq_length:])

    #Plot a comparison of true and predicted values
    data_handler.plot_comparison(true=test_y[:output_seq_length],
                                pred=pred,
                                mape=data_handler.MAPE(test_y[:output_seq_length].reshape(-1),pred),
                                title="ARIMA with lag 24 from 24-01-2009 to 31-01-2009")