# below is the implementation of resnet 50 but we can use more models 
import tensorflow
from tensorflow import keras


model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet50",
)

from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input , decode_predictions
import cv2 

image_path = "C:\\Users\\nex20\\Downloads\\archive (2)\\train\\cats\\cat_0_1631.jpeg"
img = image.load_img(image_path , target_size = (224 , 224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = preprocess_input(x)


preds = model.predict(x)
print("predicted : " , decode_predictions(preds , top=3)[0])


