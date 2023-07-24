import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.callbacks import ModelCheckpoint

train_path = "/app/data/clothing-dataset-small/train"
validation_path = "/app/data/clothing-dataset-small/validation"
test_path = "/app/data/clothing-dataset-small/test"

train_gen = ImageDataGenerator(
    # rotation_range=30,
    # width_shift_range=30.0,
    # height_shift_range=30.0,
    shear_range=10.0,
    zoom_range=0.2,
    horizontal_flip=True,
    # vertical_flip=False,
    preprocessing_function=preprocess_input
)

train_ds = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
)

validation_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_ds = validation_gen.flow_from_directory(
    validation_path,
    target_size=(299, 299),
    batch_size=32,
)


def make_model(learning_rate, droprate):
    base_model = Xception(
        weights='imagenet',
        input_shape=(299, 299, 3),
        include_top=False,
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(299, 299, 3))

    base = base_model(inputs, training=False)
    vector = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(100, activation='relu')(vector)
    drop = keras.layers.Dropout(rate=droprate)(inner)
    outputs = keras.layers.Dense(10)(drop)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    return model


def evaluate_model():
    model = keras.models.load_model('/app/xception_v1_06_0.845.h5')
    test_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_ds = test_gen.flow_from_directory(
        test_path,
        shuffle=False,
        target_size=(150, 150),
        batch_size=32,
    )

    model.evaluate(test_ds)

    img_path = "/app/data/clothing-dataset-small/test/pants/c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"

    img = load_img(img_path, target_size=(150, 150))

    # preprocess the image
    x = np.array(img)
    myX = np.array([x])
    myX = preprocess_input(myX)

    # get the prediction
    pred = model.predict(myX)
    print("pred 0:", pred[0])

    labels = {
        0: 'dress',
        1: 'hat',
        2: 'long_sleeve',
        3: 'outwear',
        4: 'pants',
        5: 'shirt',
        6: 'shoes',
        7: 'shorts',
        8: 'skirt',
        9: 't-shirt',
    }

    print(labels[pred[0].argmax()])


"""
def get_prediction(model):
    img_path = "/app/data/clothing-dataset-small/test/pants/c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"
    img = load_img(img_path, target_size=(299, 299))

    # preprocess the image
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)

    # get the prediction
    pred = model.predict(X)
    print(pred[0].argmax())

    labels = {
        0: 'dress',
        1: 'hat',
        2: 'long_sleeve',
        3: 'outwear',
        4: 'pants',
        5: 'shirt',
        6: 'shoes',
        7: 'shorts',
        8: 'skirt',
        9: 't-shirt',
    }

    print(labels[pred[0].argmax()])
"""




def main():
    checkpoint = ModelCheckpoint(
        "xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode='max',
        verbose=1
    )

    # model = make_model(learning_rate=0.001, droprate=0.2)
    # model.fit(train_ds,
    #          epochs=50,
    #          validation_data=val_ds,
    #          callbacks=[checkpoint])
    evaluate_model()

