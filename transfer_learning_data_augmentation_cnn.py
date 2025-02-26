import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical , img_to_array , array_to_img
from keras.models import load_model
import cv2
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout , BatchNormalization , Activation , InputLayer
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img



conv_layer = keras.applications.VGG16(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(256 , 256 , 3),
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
                name="vgg16",
            )
conv_layer.summary()
# other models can be used 
# finetuning 
# fine tuning 
conv_layer.trainable = False
set_trainable = False
for layer in conv_layer.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

for layer in conv_layer.layers:
    print(layer, layer.trainable)


# preparing the trainable model 

model = Sequential()
model.add(conv_layer) # adding our pretrained model 
model.add(Flatten())
model.add(Dense(256 , activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2 , activation='sigmoid'))

model.summary()


# doing same with data augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\nex20\\Downloads\\archive (2)\\train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\nex20\\Downloads\\archive (2)\\test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')


early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=0,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                    start_from_epoch=0,
                )

model.compile(optimizer=rmsprop_ , loss='binary_crossentropy' , metrics=['accuracy'])
model.fit(train_generator , epochs=10 , validation_data=validation_generator , callbacks=[early_stopping])

