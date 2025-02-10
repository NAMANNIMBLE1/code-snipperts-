model = Sequential()
model.add(Input(shape=(100,)))  # Input layer
model.add(Embedding(10000, 32))  # Embedding layer
model.add(SimpleRNN(units=32, return_sequences=True))  # RNN layer 1
model.add(SimpleRNN(units=32, return_sequences=True))  # RNN layer 2
model.add(SimpleRNN(units=32, return_sequences=True))  # RNN layer 3
model.add(SimpleRNN(units=32))  # Last RNN layer (no return_sequences)
model.add(Dense(1, activation='sigmoid'))  # Output layer

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
