Stock Price Forecast based on LSTM
=====
Welcome to the Stock Price Forcast with LSTM project. This repository contains code and resources to predict stock prices using Long Short-Term Memory (LSTM) networks. 

# LSTM Overview
## What is LSTM?
LSTM, or Long Short-Term Memory, is a type of recurrent neural network (RNN) architecture. It is designed to overcome the vanishing gradient problem in standard RNNs, making it particularly effective for sequential data, such as time series. LSTM networks are capable of capturing long-term dependencies in data, making them well-suited for stock price prediction.

LSTM cells have three gates: the input gate, the forget gate, and the output gate. These gates control the flow of information through the cell, allowing it to store and retrieve information over extended time periods. This ability to learn and remember patterns in data is what makes LSTMs powerful for time series forecasting.

More details please visit this [blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

## How LSTM Works?
LSTMs work by processing sequential data through a series of these LSTM cells. Each cell takes input data, updates its internal state, and produces an output. This output can be used for making predictions. The network is trained by adjusting the weights and biases to minimize the prediction error over a historical dataset.

## Real-Life Applications
* Investment Strategies: Stock price prediction models can be used to inform investment decisions, helping traders and investors make informed choices.

* Risk Management: Predicting stock prices can assist in assessing the risk associated with specific investments and developing risk management strategies.

* Financial Planning: Individuals can use stock price predictions to plan for their financial future, including retirement planning and saving goals.

# Ideas
* Tushare is a good resource to obtain stock data, praised for its free access. Daily stock price data and daily indicators for over 3,500 stocks were collected and classified into main board, growth board, and small and medium board. Data was merged based on stock and trading date to form the initial dataset.
* Baostack is also a great Python API, but you need to pay for it...
* Some suggestions from others(It's meaningful): 
  * According to personal needs, select the desired data set for training. For example: data with daily limit for two consecutive days is used as a piece of training data; data with daily limit for one day is used as a piece of training data; data with an increase of more than 5% for two consecutive days is used as a piece of training data. 
  * Use N days of stock data to predict N+1 day stock growth; 
  * Not every N day of stock data has a good prediction effect on N+1 day data, so we What needs to be concerned about is: N-day data with strong "expressiveness", that is, continuous daily limit, continuous increase of more than 5%, etc.; 
  * N-day data does not have obvious performance characteristics, so for data with obvious performance characteristics, it is Noise, the more noise, the louder it is, the worse it is for our predictions


# Environment
Here are some essential versions you need to pay attention to:
```
python 3.6 < x < 3.8，(I use 3.7.16)
pytorch 1.3.1，
torchvision 0.4.2 - 0.5.0 
Pillow 7.1.x,
pandas 1.0.x
CUDA: 10.1
```
As I create this project on my old laptop, so it's hard to train a larger training set for this, I only utilize one csv file for the presentation.

# Catalog
model：model-based files 
data：`csv` source data

`dataset.py`: Data loading and preprocessing class, including data normalization and splitting into training and testing sets.
`train.py`: Model training.
`evaluate.py`: Implement Model Prediction.
`LSTMModel.py`: Definition of the LSTM model.
`common_parsers.py`: Common parameters you can change.

# Getting started
- Clone this project:`git clone https://github.com/Ray7788/Stock-Price-Forecast.git`
- Train the model:`python3 train.py`
- Run the model prediction:`python3 evluteuate.py`

## Contributing
We welcome contributions from the community. Feel free to fork the repository, make improvements, and create pull requests. 
if you have questions please contact with ray778@foxmail.com
