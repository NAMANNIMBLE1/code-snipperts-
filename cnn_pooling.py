# max pooling 
import tensorflow
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1) , padding='same' , strides=(2,1)))
model.add(MaxPooling2D(pool_size=(2, 2) , padding='same' , strides=(2,1)))
model.add(Conv2D(32, (3, 3), activation='relu' , padding='same' , strides=(2,1)))
model.add(MaxPooling2D(pool_size=(2, 2) , padding='same' , strides=(2,1)))
model.add(Dropout(0.25))    

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= 1 , batch_size= 100 , verbose= 1)   
model.evaluate(X_test, y_test)
