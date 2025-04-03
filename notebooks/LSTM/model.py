import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import InputLayer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from preprocessing import data_scaling


def build_lstm_model(sequence_length, input_dim):
    """Builds an LSTM model."""
    model = Sequential([
        InputLayer((sequence_length, input_dim)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model