import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from tensorflow.python.keras.callbacks import History

from ..helpers.functions import (
    create_dataset,
)


class FeatExtract:
    def __init__(self):
        self._conv_base = keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
        )
        self._conv_base.trainable = False

    def make_model(self) -> keras.Model:
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ]
        )

        inputs = keras.Input(shape=(180, 180, 3))
        x = data_augmentation(inputs)
        x = keras.applications.vgg16.preprocess_input(x)
        x = self._conv_base(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=256)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        return keras.Model(inputs, outputs)

    def train_fit(self, model: keras.Model) -> History:
        model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="/app/models/feature_extraction_with_augmentation{epoch:02d}_{val_loss:.3f}.krs",
                save_best_only=True,
                monitor="val_loss",
            )
        ]
        history = model.fit(
            create_dataset("train"),
            epochs=50,
            validation_data=create_dataset("validation"),
            callbacks=callbacks,
        )
        return history

    def plot_result(self, history: History) -> None:
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

    def test_model(self):
        test_model = keras.models.load_model(
            "/app/models/feature_extraction_with_augmentation37_1.506.krs"
        )
        test_loss, test_acc = test_model.evaluate(create_dataset("test"))
        print(f"Test accuracy: {test_acc:.3f}")


def run():
    feat_extract = FeatExtract()
    # model: keras.Model = feat_extract.make_model()
    # history: History = feat_extract.train_fit(model=model)
    # feat_extract.plot_result(history=history)
    feat_extract.test_model()
