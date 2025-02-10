# now we will use lstm model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input

model = Sequential()
model.add(Input(shape=(56,)))
model.add(Embedding(283, 100))
model.add(LSTM(150))
model.add(Dense(283, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
