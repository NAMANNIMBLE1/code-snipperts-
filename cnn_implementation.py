import tensorflow
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D # there are 2 type of pooling min , average and maximum 
from keras.datasets import mnist


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1) , padding='same')) # padding are 2 valid and same 
model.add(Conv2D(32, (3, 3), activation='relu' , padding='same')) # padding are 2 valid and same 
model.add(Conv2D(32, (3, 3), activation='relu' , padding='same')) # padding are 2 valid and same 

model.add(Flatten())
model.add(Dense(128, activation='relu'))    
model.add(Dense(10, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=1)

model.evaluate(X_test, y_test)
