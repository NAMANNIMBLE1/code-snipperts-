import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout , RNN , SimpleRNN , Input , Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Input(shape=(100,)))  # Input layer
model.add(Embedding(input_dim=10000, output_dim=32))  # Embedding layer
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))  # First BiLSTM
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))  # Second BiLSTM
model.add(Bidirectional(LSTM(units=32, return_sequences=False)))  # Last BiLSTM, return_sequences=False
model.add(Dense(1, activation='sigmoid'))  # Output layer

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
