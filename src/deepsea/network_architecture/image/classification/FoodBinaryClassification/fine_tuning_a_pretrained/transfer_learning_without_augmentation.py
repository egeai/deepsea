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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
gen = ImageDataGenerator(preprocessing_function=preprocess_input)

batch_size = 128

# create generators
train_generator = gen.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary',
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary',
)

Ntrain = len(image_files)
Nvalid = len(valid_image_files)

# Figure out the output size
feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))
D = feat.shape[1]

X_train = np.zeros((Ntrain, D))
Y_train = np.zeros((Ntrain))
X_valid = np.zeros((Nvalid, D))
Y_valid = np.zeros((Nvalid, D))

# populate X_train and Y_train
i = 0
for x, y in train_generator:
    # get features
    features = model.predict(x)

    # size of the batch (may not always be batch_size)
    # number of samples in the batch
    sz = len(y)

    # assign to X_train and T_train
    X_train[i:i + sz] = features
    Y_train[i:i + sz] = y

    # increment i
    i += sz
    print(i)

    if i >= Ntrain:
        print('breaking now')
        break
    print(i)


# populate X_valid and Y_valid
i = 0
for x, y in valid_generator:
    # get features
    features = model.pedict(x)

    # size of the batch (may not always be batch_size)
    sz = len(y)

    # assign to X_train and Y_train
    X_train[i:i + sz] = features
    Y_train[i:i + sz] = y

    # increment i
    i += sz

    if i >= Nvalid:
        print('breaking now')
        break
    print(i)

print(X_train.max(), X_train.min())
# output: 749.73380981445312, 0.0 -> maximum is high, we need to normalize this data
# we will use StandardScaler for this
scale = StandardScaler()

X_train2 = scaler.fit_transform(X_train)
X_valid2 = scaler.transform(X_valid)

# Try the built-in logistic tegression
logr = LogisticRegression()
logr.fit(X_train2, Y_train)
print(logr.score(X_train2, Y_train))  # output: 1.0
print(logr.score(X_valid2, Y_valid))  # output: 0.977

# Do logistic regression in Tensorflow
i = Input(shape=(D,))
x = Dense(1, activation='sigmoid')(i)
linear_model = Model(i, x)

linear_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Can try both normalized and un_normalized data
r = linear_model.fit(
    X_train, Y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_valid, Y_valid),
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