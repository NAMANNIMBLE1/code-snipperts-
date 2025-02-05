import tensorflow as tf
import keras
from keras import layers

callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=0,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0,
)

model = keras.Sequential(
    [
        layers.Dense(10, activation="relu", name="layer1"),
        layers.Dense(5, activation="relu", name="layer2"),
        layers.Dense(1, activation="sigmoid", name="layer3"),
    ]
)

model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])

history = model.fit(X_train , y_train , validation_data=(X_test , y_test) , epochs= 3400 , verbose= 0 ,callbacks=[callback])


import matplotlib.pyplot as plt 

plt.plot(history.history['loss'], label = "train")
plt.plot(history.history['val_loss'] , label = "test")
plt.legend()
plt.show()
