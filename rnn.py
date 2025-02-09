
### RNN Implementation using Keras
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras import layers

model = Sequential()
model.add(layers.SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)))
model.add(layers.Dense(1, activation='linear'))  # Output a continuous value
model.compile(optimizer='adam', loss='mean_squared_error')
return model

# Train the model
model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X)

