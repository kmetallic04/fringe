#Total length of the sequence we want to use
#We set the sequence length to 168 since we want to work with a one-week interval
seq_length = 168
output_seq_length = 168
total_sequence_length = 33600 + output_seq_length
#Univariate time series, input and output dimensions are 1
input_dim=1
output_dim=1
#Small batch size to save on memory and take on smaller gradient steps
batch_size = 16
#Since we'll be using a generator for samples
steps_per_epoch = 100
#Chosen empirically
learning_rate = 1e-3

#Set the arima variables and bundle them in a tuple
p = 24
d = 1
q = 2

order = (p,d,q)