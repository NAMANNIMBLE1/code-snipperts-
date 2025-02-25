from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array , array_to_img


image = image.load_img("your path" , target_size=(256,256))
plt.imshow(image)

datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.2,1.0]
        )

image = img_to_array(image)
# image.shape()
input_batch = image.reshape(1 , 256 , 256 , 3) # reshaping for batches



i = 0
for batch in datagen.flow(input_batch , batch_size=1 , save_to_dir='directory_to_store' , save_prefix='cat' , save_format='jpeg'):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i == 4: # how much images to generate
        break

# proper use in projects #################################################################################



train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.2,1.0],
            rescale=1./255
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    directory="C:\\Users\\nex20\\Downloads\\archive (2)\\train",
                    target_size=(256,256),
                    color_mode='rgb',
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True,
                    seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='jpeg',
                    follow_links=False,
                    subset=None,
                    interpolation='nearest'
                )


validation_generator = test_datagen.flow_from_directory(
                    directory="C:\\Users\\nex20\\Downloads\\archive (2)\\test",
                    target_size=(256,256),
                    color_mode='rgb',
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True,
                    seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='jpeg',
                    follow_links=False,
                    subset=None,
                    interpolation='nearest'
                )

model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
model.fit(train_generator , validation_data=validation_generator , epochs=100 , batch_size=32 , callbacks= [early_stopping])



