import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.03)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.03)),
    Dense(128, activation='sigmoid', kernel_regularizer=l2(0.03))
])


# for images cnn 
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Example for 28x28 images
    Dense(128, activation='relu', kernel_regularizer=l2(0.03)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.03)),
    Dense(128, activation='sigmoid', kernel_regularizer=l2(0.03))
])

