# A comparison of ARIMA v RNN (GRU) on real-valued time series predictions

This project compares the accuracy of an Autoregressive Integrated Moving Average model and a Recurrent Neural Network based on Gated Recurrent Units (GRUs) with regards to real-valued time series prediction. The input will be a one week sequence of national hourly electric load demand data for Austria, and the output expected is a prediction of the following week's hourly load demand data. This approximation is solely based on a real-valued time series analysis of historical data.

For the Recurrent Neural Network:
To train the model, run:

```python recurrent.py train```

This will train the RNN on the entire training data. Internally, the function shuffles different 168-hour-long sequences to increase the entropy of the training data.

For testing:

```python recurrent.py test```


Testing simply shows a matplotlib graph of the true and predicted sequences plotted on the same axes for comparison. Using the ```test``` evaluates the immediate one-week sequence following the training data.

For the ARIMA model:

To fit the model, run:

```python time_series.py train```

To test:

```python time_series.py test```


To calculate the average Mean Absolute Percentage Error on all test data:

```python recurrent.py mape```

or

```python time_series.py mape```


The following graphs show the resultant predictions against the targets for the neural nets:

![256 recurrent units](https://github.com/kmetallic04/fringe/blob/master/images/1.png)

![128 recurrent units](https://github.com/kmetallic04/fringe/blob/master/images/2.png)

![64 recurrent units](https://github.com/kmetallic04/fringe/blob/master/images/3.png)

The graph of the sample test of the autoregressive method is show below:

![ARIMA 24 lag](https://github.com/kmetallic04/fringe/blob/master/images/4.png)


Autoregressive methods on average perform better than neural nets. The extra features offered by AI seem to be useful in tasks such as language processing. Equations are still better at math than neural nets.
