import tensorflow
from tensorflow import keras 
from keras import layers


model = keras.Sequential(
    [
        layers.Dense(255 , activation='relu' , name = "layer 1"),
        layers.Dropout(0.25),
        layers.Dense(255 , activation='relu' , name = "layer 2"),
        layers.Dropout(0.25),
        layers.Dense(1 , activation='sigmoid', name = "layer 3")
    ]
)

model.compile(optimizer = 'Adam' , loss = "mse" , metrics = ['accuracy'])
model.fit(X_train , y_train , epochs = 600 , validation_data = (X_test , y_test)  , verbose = False)
