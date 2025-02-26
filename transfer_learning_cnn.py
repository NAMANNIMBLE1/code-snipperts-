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


# getting data without data augmentation 
from keras import utils

train_data = utils.image_dataset_from_directory(
                directory="C:\\Users\\nex20\\Downloads\\archive (2)\\train",
                labels='inferred',
                label_mode='categorical',
                class_names=None,
                color_mode='rgb',
                batch_size=32,
                image_size=(256, 256),
                shuffle=True,
                seed=None,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )

test_data = utils.image_dataset_from_directory(
                directory="C:\\Users\\nex20\\Downloads\\archive (2)\\test",
                labels='inferred',
                label_mode='categorical',
                class_names=None,
                color_mode='rgb',
                batch_size=32,
                image_size=(256, 256),
                shuffle=True,
                seed=None,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )

train_data.class_names


# when using the utils for getting the data 

# normalizing 
def process(image , label):
    return tf.cast(image/255.0 , tf.float32) , label

train_data = train_data.map(process)
test_data = test_data.map(process)


rmsprop_ = keras.optimizers.RMSprop(learning_rate=1e-5)
model.compile(optimizer=rmsprop_ , loss='binary_crossentropy' , metrics=['accuracy'])
model.fit(train_data , epochs=10 , validation_data=test_data)









