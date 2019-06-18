#Ensuring compatibility between python 2.7 and 3.x...
from __future__ import unicode_literals, print_function, division

import sys
import time
from time import time as time_

import tensorflow as tf

#For handling numbers
import numpy as np

#For performing inverse transform
from sklearn.preprocessing import StandardScaler

#For graphical plot display
import matplotlib.pyplot as plt 
from matplotlib import style

import data_handler
import config

#Paths to saved training weights
path_to_checkpoint = './checkpoints_delta/gru_256_bs_16_sl_168_ep_100'

epochs = 100

#Importing hyperparameters
total_sequence_length = config.total_sequence_length
input_dim=config.input_dim
output_dim=config.output_dim
seq_length = config.seq_length
output_seq_length = config.output_seq_length
batch_size = config.batch_size
steps_per_epoch = config.steps_per_epoch
learning_rate = config.learning_rate

def build_model():
    #Define the model
    model = tf.keras.models.Sequential()
    #Add first GRU layer
    model.add(tf.keras.layers.GRU(256, activation='relu', return_sequences=True, input_shape=(seq_length, input_dim)))
    #Define an initializer to initialize weights
    initializer = tf.keras.initializers.glorot_uniform()
    #Add the second single layer perceptron
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer))
    #Output layer (third)
    model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer))
    #Define adam optimizer
    optimizer = tf.keras.optimizers.Adadelta()
    #Compile model with mse loss
    model.compile(loss='mse', optimizer=optimizer)

    checkpoints=tf.keras.callbacks.ModelCheckpoint(
        filepath=path_to_checkpoint,
        monitor='loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True
    )

    #For graphical report of training progress
    logs=tf.keras.callbacks.TensorBoard(
        log_dir='logs_delta/{}'.format(time_()),
        histogram_freq=0,
        write_graph=False
    )

    #Reduces the learning rate if the loss is stuck on local minima
    reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        min_lr=1e-5,
        patience=0,
        verbose=1
    )

    #Early stopping regularization strategy
    early_stopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
    )


    #List of functions called at end of each training epoch
    callbacks=[checkpoints,logs,reduce_lr,early_stopping]

    return model, callbacks

#A function to predict data
def predict(model, data, label_scaler):
    data_x = data[:seq_length]
    data_x = np.expand_dims(data_x, axis=0)
    pred_y = model.predict(data_x)
    pred_y = label_scaler.inverse_transform(pred_y)

    return pred_y.reshape(output_seq_length,1)

raw_x, raw_y = data_handler.fetch_data()
#Standardizing data to force it to lie on the same range
scaler_x, x, scaler_y, y = data_handler.standardize_data(raw_x, raw_y)
#Splitting data into training and testing sets
train_x, test_x = data_handler.split_data(x)
train_y, test_y = data_handler.split_data(y)
#Define validation data as the entire training set
validation_data = (test_x.reshape(-1,seq_length,1), test_y.reshape(-1,output_seq_length,1))
#Build model
model, callbacks = build_model()
#Accept inputs from command line
train = sys.argv[1]

if (train == 'train'):
    #Call the input generator to yield a batch of sequences
    input_gen = data_handler.generate_input_pipe(train_x, train_y,batch_size,seq_length, output_seq_length)
    #Train the model
    model.fit_generator(generator=input_gen,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch, 
                        callbacks=callbacks,
                        validation_data=validation_data)
elif(train == 'test'):
    #Load the stored parameters (depending on the model architecture)
    model.load_weights(path_to_checkpoint)
    #First sequence starts at zero for the demonstration case
    #This value is the index for the beginning of the week you want to test among those in the test data
    #Can be, for example, a random integer that is a multiple of 168 but within the range of test data
    #Can also be a function of your choice
    spot = 0
    _ , raw_test_y = data_handler.split_data(raw_y)
    raw_test_y = raw_test_y[spot:]
    pred = predict(model, test_x[spot:], scaler_y)
    #Compare actual and predicted values
    data_handler.plot_comparison(true=raw_test_y[:output_seq_length],
                                pred=pred,
                                mape=data_handler.MAPE(raw_test_y[:output_seq_length],pred),
                                title="RNN - 3 layers // {} hidden recurrent units from 24-01-2009 to 31-01-2009")
elif(train == 'mape'):
    #Load model weights
    model.load_weights(path_to_checkpoint)
    _ , raw_test_y = data_handler.split_data(raw_y)
    mape_sum = 0
    num_test_seq = len(test_x)//output_seq_length
    #Loop through test data to add the MAPE of each week
    for i in range(num_test_seq):
        spot = i * output_seq_length
        raw_test_y_ = raw_test_y[spot:]
        pred = predict(model, test_x[spot:], scaler_y)
        mape_sum += data_handler.MAPE(raw_test_y_[:output_seq_length], pred)
    #Divide by number of sequences to get average MAPE
    mape_avg = mape_sum/num_test_seq
    print(mape_avg)