"""
Source Code:
Title: Deep Learning - Convolutional Neural Networks with TensorFlow
Publisher: Packt
By Lazy Programmer
"""

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, \
    preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os


"""
Data from: https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/
wget --passive-ftp --prefer-family=ipv4 --ftp-user datasets@mmspgdata.epfl.ch --ftp-password ohsh9jah4T -nc \
ftp://tremplin.epfl.ch/FoodImage/Food-5K.zip
-------------------------------------------
# look at an image
# plt.imshow(image.load_img('../../../../../../data/Food-5K/training/0_808.jpg'))
# plt.show()

# Food images start with 1, non-food images start with 0
# plt.imshow(image.load_img('../../../../../../data/Food-5K/training/1_616.jpg'))
# plt.show()
---------------------------------------------------------
# Make directories to store the data Keras-style
!mkdir data/train
!mkdir data/test
!mkdir data/train/nonfood
!mkdir data/train/food
!mkdir data/test/nonfood
!mkdir data/test/food
-------------------------------------------------------
# Move the images
#Note: we will consider 'training' to be the train set
    'validation' folder will be the test set
# ignore the 'evaluation' set
!mv training/0*.jpg data/train/nonfood
!mv training/1*.jpg data/train/food
!mv validation/0*.jpg data/test/nonfood
!mv validation/1*.jpg data/test/food
"""

train_path='../../../../../../data/train'
valid_path='../../../../../../data/test'

# These images are pretty big and of different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200, 200]

# useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')

# useful for getting number of classes
folders = glob(train_path + '/*')

print("classes: ", folders)

# Look at the image
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],  # [3] is number of color channels
    weights='imagenet',
    include_top=False)

# freeze pretrained model weights
ptm.trainable = False


# map the data into feature vectors

# Keras image data generator returns classes one-hot encoded

K = len(folders) # number of classes
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x)

# create a model object
model = Model(inputs=ptm.input, outputs=x)

# view the structure of the model
model.summary()

# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift=0.1,
    height_shift=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horzontal_flip=True,
    preprocessing_function=preprocess_input
)

batch_size = 128

# create generators
train_generator = gen.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# fit the model
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    steps_per_epoch=int(np.ceil(len(image_files) / batch_size)),
    validation_steps=int(np.ceil(len(valid_image_files) / batch_size)),
)

# loss
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train_acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()