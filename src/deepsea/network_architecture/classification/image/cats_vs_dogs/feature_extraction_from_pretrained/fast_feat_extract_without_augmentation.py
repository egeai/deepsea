import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

from ..helpers.functions import (
    create_dataset,
)

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet", include_top=False, input_shape=(180, 180, 3)
)


def _get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


def _make_model():
    inputs = keras.Input(shape=(5, 5, 512))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


def _train_fit(model, train_features, train_labels, val_features, val_labels):
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="/app/models/feture_extraction_{epoch:02d}_{val_loss:.3f}.krs",
            save_best_only=True,
            monitor="val_loss",
        )
    ]
    history = model.fit(
        train_features,
        train_labels,
        epochs=20,
        validation_data=(val_features, val_labels),
        callbacks=callbacks,
    )
    return history


def _plot(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def run():
    train_dataset = create_dataset("train")
    val_dataset = create_dataset("validation")
    test_dataset = create_dataset("test")

    train_features, train_labels = _get_features_and_labels(train_dataset)
    val_features, val_labels = _get_features_and_labels(val_dataset)
    test_features, test_labels = _get_features_and_labels(test_dataset)

    model = _make_model()

    history = _train_fit(model, train_features, train_labels, val_features, val_labels)

    _plot(history=history)
