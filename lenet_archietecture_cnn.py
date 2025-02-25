from tensorflow import  keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout , Input , AveragePooling2D
from keras.models import Model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/255
X_test = X_test/255


model = keras.Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2) , strides=2 , padding='valid'))

model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh' , padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2) , strides=2 , padding = 'valid'))

model.add(Flatten())


model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= 1 , batch_size= 100 , verbose= 1)
model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
print("predicted values is {}".format(predictions[0]))
print("actual values is : {}".format(X_test[0]))


import matplotlib.pyplot as plt

plt.imshow(X_test[100], cmap='gray')
plt.show()
plt.imshow(predictions[100].reshape(1,10), cmap='gray')
plt.show()

