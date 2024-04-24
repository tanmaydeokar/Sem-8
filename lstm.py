import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import time

# Function to download historical stock price data
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock price data
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train LSTM model
def train_lstm_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, batch_size=1, epochs=epochs)

# Function to predict stock prices
def predict_stock_prices(model, scaler, scaled_data):
    inputs = scaled_data[len(scaled_data) - 60:]
    X_test = []
    X_test.append(inputs)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices


def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


    # Streamlit app
def main():
    st.title("Stock Price Prediction with LSTM")

    # User input for stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

    # Button to train and predict
    if st.button("Train and Predict"):
        # Download historical stock price data
        end_date = datetime.datetime.now()
        start_date = "2016-01-01"
        stock_data = download_stock_data(stock_symbol, start_date, end_date)

        # Preprocess data
        scaled_data, scaler = preprocess_data(stock_data)

        # Prepare training data
        X_train = []
        y_train = []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Train LSTM model
        lstm_model = create_lstm_model()
        start_time = time.time()
        train_lstm_model(lstm_model, X_train, y_train, epochs=1)
        end_time = time.time()
        training_time = end_time - start_time

        # Predict stock prices
        predicted_prices = predict_stock_prices(lstm_model, scaler, scaled_data)
        predicted_price_float = float(predicted_prices)
        actual_prices = scaler.inverse_transform(scaled_data)
        rmse = compute_rmse(actual_prices[-len(predicted_prices):], predicted_prices)
        # Display historical and predicted stock prices
        st.subheader("Historical Stock Prices")
        st.line_chart(stock_data["Close"])
        st.subheader("RMSE ")
        st.success(rmse)
        st.subheader("Predicted Price for Tomorrow is ")
        st.success(predicted_price_float)
        
        st.subheader("Training Time for the LSTM Model (in sec) ")
        st.success(training_time)
      

      

if __name__ == "__main__":
    main()


