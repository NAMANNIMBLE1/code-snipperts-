import tensorflow 
from tensorflow import keras
from keras.models import Model
from keras.layers import *


input_layer1 = Input(shape=(3,))
input_layer2 = Input(shape=(128,))

# First branch
hidden1 = Dense(128, activation='relu')(input_layer1)
hidden2 = Dense(128, activation='relu')(hidden1)
output1 = Dense(1, activation='sigmoid')(hidden2)
output2 = Dense(1, activation='sigmoid')(hidden2)

# Second branch
hidden3 = Dense(128, activation='relu')(input_layer2)
hidden4 = Dense(128, activation='relu')(hidden3)
output3 = Dense(1, activation='sigmoid')(hidden4)
output4 = Dense(1, activation='sigmoid')(hidden4)

# Concatenation of outputs
combined = Concatenate()([output1, output2, output3, output4])

# Fully connected layers after concatenation
z = Dense(128, activation='relu')(combined)
z = Dense(128, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)


model = Model(inputs=[input_layer1, input_layer2], outputs=z)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')
