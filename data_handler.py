#Ensuring compatibility between python 2.7 and 3.x...
from __future__ import unicode_literals, print_function, division


#For handling numbers and datasets respectively
import numpy as np
import pandas as pd

#For plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#For standardizing data
from sklearn.preprocessing import StandardScaler

import config

#importing hyperparameters
total_sequence_length = config.total_sequence_length
input_dim=config.input_dim
output_dim=config.output_dim
seq_length = config.seq_length
output_seq_length = config.output_seq_length
batch_size = config.batch_size

#Create a parser to 'decode' the dates
dateparser = lambda x : pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
path_to_dataset = './datasets/time_series_60min_singleindex.csv'
dataset = pd.read_csv(path_to_dataset,parse_dates=['utc_timestamp'],date_parser=dateparser,dtype={'AT_load_entsoe_power_statistics':np.float32},low_memory=False)

#Verify alignment of data
assert len(dataset.loc[:,'utc_timestamp']) == len(dataset.loc[:,'AT_load_entsoe_power_statistics'])

input_df = pd.DataFrame(dataset.loc[1:total_sequence_length,'utc_timestamp'])
#Rename columns appropriately
input_df.rename(columns={'utc_timestamp':'utc_time'}, inplace=True)
input_df.loc[:,'load_demand'] = pd.Series(dataset.loc[1:total_sequence_length,'AT_load_entsoe_power_statistics'], index=input_df.index)
#Enforce indexing according to the date/time
input_df.set_index('utc_time', inplace=True)
#print(np.shape(input_df))
#We shift the dataframe upward by a week to enable us to find the exact next sequence in a function below
#Our target dataframe. Not all elements of this array will be used.
target_df = input_df.shift(-seq_length)
#Converting to immutable numpy arrays, easier to work with
inputs = np.array(input_df.loc[:,'load_demand'])[:-seq_length]
targets = np.array(target_df.loc[:,'load_demand'])[:-seq_length]

def fetch_data():
    #Reshaping to conform to sklearn's accepted shapes
    return inputs.reshape(-1,1), targets.reshape(-1,1)

def standardize_data(inputs, targets):
    #Scalers are returned for inverse transform on preds
    inputs_scaler = StandardScaler()
    inputs_scaled = inputs_scaler.fit_transform(inputs)
    targets_scaler = StandardScaler()
    targets_scaled = targets_scaler.fit_transform(targets)
    return inputs_scaler, inputs_scaled, targets_scaler, targets_scaled

#Define a function to split between train and test data
#80% for training, 20% for testing
def split_data(data):
    return (data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):])


#Define a function that generates inputs and test data favourable for our model
def generate_input_pipe(inputs, targets, batch_size, seq_length, output_seq_length):
    #Infinite loop
    while True:
        #Allocate an array for input batches
        input_shape = (batch_size, seq_length, input_dim)
        input_batch = np.zeros(dtype=np.float32,shape=input_shape)

        #Alllocate an array for targets
        target_shape = (batch_size, output_seq_length, output_dim)
        target_batch = np.zeros(dtype=np.float32,shape=target_shape)


        for i in range(batch_size):
                # Get a random start-index.
                # This points to a random point in the training-data. This simulates shuffling
                #The random nteger is muultiplied by 24 to align with the beginning of day
                idx = np.random.randint(700) * 24

                # Copy the sequences of data starting at this index.
                input_batch[i] = inputs[idx:(idx+seq_length)]
                target_batch[i] = targets[idx:(idx+output_seq_length)]

        yield input_batch, target_batch

#Draws the autocorrelation graph for analysing the lag for ARIMA
def show_autocorrelation():
    load_demand_values = input_df.loc[:,'load_demand']
    autoc_plot = pd.plotting.autocorrelation_plot(load_demand_values[:1000])
    autoc_plot.axvline(12,c='r',linewidth=1.5)
    plt.title("Autocorrelation Plot")
    plt.show()
    return 0

#Performs differencing
def remove_seasonality(series, lag):
    diff = list()
    for i in range(lag, len(series)):
        value = series[i] - series[i - lag]
        diff.append(value)
    series = pd.Series(diff)
    return series

#Adds seasonality back to restore real values
def add_seasonality(series, predictions, lag=1):
    return predictions + series[-lag]

#Calculates MAPE against two real-valued arrays
def MAPE(test_y, pred):
    mape_score = (np.sum(np.abs(test_y - pred) / test_y) / len(test_y)) *100
    return round(mape_score, 3)

#Plots an actual v predicted graph
def plot_comparison(true, pred, mape, title):
    plt.figure(figsize=(15,5))
    plt.plot(true, label='Future (target) values')
    plt.plot(pred, label='Future (predicted) values')
    plt.grid()
    plt.title("{} (MAPE = {}%)".format(title, mape))
    plt.xlabel("Hours into week")
    plt.ylabel("Load demand values")
    plt.legend()
    plt.show()