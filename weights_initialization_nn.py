import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os

# Define Model
model = Sequential([
    Dense(10, activation='tanh', input_shape=(2,)),  # Input Layer
    Dense(10, activation='tanh'),                     # Hidden Layer
    Dense(1, activation='sigmoid')                    # Output Layer
])

# Display Model Summary
model.summary()

# Get Initial Weights
initial_weights = model.get_weights()

# Set New Random Weights with Correct Shapes
initial_weights[0] = np.random.randn(2, 10) * np.sqrt(1 / 2)   # Weights of first layer
initial_weights[1] = np.zeros(10)                              # Bias of first layer
initial_weights[2] = np.random.randn(10, 10) * np.sqrt(1 / 10) # Weights of second layer
initial_weights[3] = np.zeros(10)                              # Bias of second layer
initial_weights[4] = np.random.randn(10, 1) * np.sqrt(1 / 10)  # Weights of output layer
initial_weights[5] = np.zeros(1)                               # Bias of output layer

# Set the modified weights to the model
model.set_weights(initial_weights)

# Verify Updated Weights
print(model.get_weights())


