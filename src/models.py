import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from typing import Tuple

def build_lstm_model(
        input_shape: Tuple[int, int] = (30, 7),
        lstm_units: list = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 5e-4
) -> tf.keras.Model:
    ''' Constructs and compiles a 2-layer Stacked LSTM Binary Classifier.'''

    model = Sequential([
        # Layer 1: LSTM with 64 units returning full sequences
        LSTM(
            lstm_units[0],
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=12(1e-4) #L2 regularization penalizes oversized weights
        ),
        Dropout(dropout_rate),

        LSTM(
            lstm_units[1],
            return_sequences=False,
            kernel_regularizer=l2(1e-4)
        ),
        Dropout(dropout_rate),

        # Output Layer: 1 sigmoid neuron for binary probability P(UP)
        Dense(1, activation='sigmoid')
    ])

    # Compile with Binary Cross-Entropy and Adam Optimizer
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model