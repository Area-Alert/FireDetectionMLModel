import keras

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier=Sequential()
#convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening


classifier.add(Flatten())
#Hidden and output layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset\\train_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
from PIL import Image
classifier.fit_generator(
        train_set,
        steps_per_epoch=100,
        epochs=15,
        validation_data=test_set,
        nb_val_samples=500,
        use_multiprocessing=False,
        workers=8)

train_set[0]
import numpy as np

from keras.preprocessing import image
test_image = image.load_img('dataset//single_prediction//1200px-Good_Food_Display_-_NCI_Visuals_Online.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    print("Not Fire")
else:
    print("Fire")

print(result)

test_image = image.load_img('dataset//single_prediction//A-Alamy-BXWK5E_vvmkuf.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print(result)