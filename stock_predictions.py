# Import all necessary python libraries
import datetime
import sklearn
import keras
import tensorflow
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Set date format for time series Analysis
today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

# Let's import the data from yahoo finance and index Date column

data = yf.download('APL', start=start_date, end=end_date, progress=False)
data['Date'] = data.index
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
data.reset_index(drop=True, inplace=True)
data.tail()

# Here we Plot imported Data on a candle stick chart to give clear picture of increase and decrease.

figure = go.Figure(
    data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
figure.update_layout(title="Apple stock price analysis", xaxis_rangeslider_visible=False)
figure.show()

# Let's see the correlation of all columns with the 'close' column being the largest.
correlation = data.corr()
print(correlation['Close'].sort_values(ascending=False))

# Train LSTM for stock Price Prediction
x = data[['Open', 'High', 'Low', 'Volume']]
y = data[['Close']]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Now let's prepare a neural network architecture for LSTM

from keras.models import Sequential

from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# Here is how to train your neural network model for stock price prediction

model.compile(Optimizer='adam', Loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)

# Now let's test this model by giving inpute values
# note that features =[Open, high, Low, Adj close, Volume]

features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
model.predict(features)
