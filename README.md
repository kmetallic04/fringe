# Time_Series_v_RNN_GRU
This project compares an Autoregressive Integrated Moving Average model and a Recurrent Neural Network based on Gated Recurrent Units (GRUs) on time series analysis. The input will be a one week sequence of national hourly load demand data for Austria, and the output expected is a prediction of the next week's hourly load demand data. This approximation is purely based on a real-valued time series analysis of historical data. Python 3.6 is required.

For the Recurrent Neural Network:
To train the model, run:

```python recurrent.py train```
For testing:

```python recurrent.py test```

For the ARIMA model:
To fit the model, run:

```python time_series.py train```
To test:

```python time_series.py test```

Testing using the ```test``` argument is done on the immediate sequence following the test data.
To calculate the average Mean Absolute Percentage Error on all test data:

```python recurrent.py mape```
or

```python time_series.py mape```
