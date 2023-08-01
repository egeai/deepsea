import matplotlib.pyplot as plt
import keras

from src.deepsea.network_architecture.classification.image.cats_vs_dogs.helpers import (
    functions,
)
from src.deepsea.network_architecture.classification.image.cats_vs_dogs.basic.convnet_model import (
    make_model,
)


def cats_vs_dogs():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="/app/models/{epoch:02d}_{val_loss:.3f}.k",
            save_best_only=True,
            monitor="val_loss",
        )
    ]

    train_dataset = functions.create_dataset("train")
    validation_dataset = functions.create_dataset("validation")

    model = make_model()

    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    # Plot the loss and accuracy of the model over training and validation data
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def evaluate():
    test_model = keras.models.load_model("/app/models/14_0.485.k")
    test_dataset = functions.create_dataset("test")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
