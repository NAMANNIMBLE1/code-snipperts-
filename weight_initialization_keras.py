import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define Model
model = Sequential([
    Dense(10, activation='tanh', input_shape=(2,) , kernel_initializer= 'he_normal'),  # Correct input shape
    Dense(10, activation='tanh' ,kernel_initializer= 'he_normal'),
    Dense(1, activation='sigmoid', kernel_initializer= 'he_normal')
])

# Display Model Summary
model.summary()
