import tensorflow as tf
from tensorflow import keras
from keras import model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=2))  # Added units
model.add(BatchNormalization())
model.add(Dense(units=32, activation='relu'))  # Defined units
model.add(Dense(units=2, activation='sigmoid'))  # Output layer with units

print(model.summary())
