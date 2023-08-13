import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.src.callbacks import History
from tensorflow.keras.layers import Embedding
from keras.layers import (
    Input,
    Activation,
    Dense,
    Permute,
    Dropout,
    add,
    dot,
    concatenate,
    LSTM,
)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tf.kears.layers import TextVectorization


class SimpleChatBot:
    def __init__(self):
        """
        Length of train data is 10.000
        Sample train data:
        (['Mary', 'moved', 'to', 'the', 'bathroom', '.', 'Sandra', 'journeyed', 'to', 'the', 'bedroom'], -> Story
        ['Is', 'Sandra', 'in', 'the', 'hallway', '?'], -> Question
        'no') -> Answer
        """
        self.tokenizer = None
        self.vocab = set()

        self.train_story_text: list = []
        self.train_question_text: list = []
        self.train_answers: list = []
        self.all_data: list = []
        self.train_story_seq: list[list] = []

        self.max_story_length: int = 0
        self.max_question_length: int = 0

        self.vocab_len: int
        with open(
            "app/data/Advanced-Chatbots-with-Deep-Learning-Python/Data/train_qa.txt",
            "rb",
        ) as f:
            self.train_data = pickle.load(f)
        # Length of test data is 1.000
        with open(
            "app/data/Advanced-Chatbots-with-Deep-Learning-Python/Data/test_qa.txt",
            "rb",
        ) as f:
            self.test_data = pickle.load(f)
        self.all_data = self.test_data + self.train_data

    def make_vocabulary(self) -> None:
        for story, question, answer in self.all_data:
            self.vocab = self.vocab.union(set(story))
            self.vocab = self.vocab.union(set(question))
        # add 'yes' and 'no' answers to the vocab
        self.vocab.add("yes")
        self.vocab.add("no")
        # get vocabulary length
        vocab_length = len(self.vocab) + 1

    def max_story_question_len(self):
        self.max_story_length = max([len(data[0]) for data in self.all_data])
        self.max_question_length = max([len(data[1]) for data in self.all_data])

    def create_tokenizer_sequence(self) -> None:
        # create tokenizer with no filter
        self.tokenizer = Tokenizer(filters=[])
        self.tokenizer.fit_on_texts(self.vocab)
        # we can see word_index of the tokenizer
        # it's like {'bathroom':1, 'in': 2, 'grabbed': 3, 'no': 18, 'yes': 31, ...}
        for story, question, answer in self.train_data:
            self.train_story_text.append(story)
            self.train_question_text.append(question)
            self.train_answers.append(answer)
        self.train_story_seq = self.tokenizer.texts_to_sequences(self.train_story_text)

    def vectorize_stories(self, data):
        X, Xq, Y = []
        for story, query, answer in data:
            x_story_indices = [
                self.tokenizer.word_index[word.lower()] for word in story
            ]
            x_query_indices = [
                self.tokenizer.word_index[word.lower()] for word in query
            ]
            y = np.zeros(len(self.tokenizer.word_index) + 1)
            y[self.tokenizer.word_index[answer]] = 1

            X.append(x_story_indices)
            Xq.append(x_query_indices)
            Y.append(y)
        return (
            pad_sequences(X, maxlen=self.max_story_length),
            pad_sequences(Xq, maxlen=self.max_question_length),
            np.array(Y),
        )

    def model(self):
        # Vectorize train and test
        inputs_train, queries_train, answer_train = self.vectorize_stories(
            self.train_data
        )
        inputs_test, queries_test, answer_test = self.vectorize_stories(self.test_data)

        input_sequence = Input((self.max_story_length,))
        question = Input((self.max_question_length,))

        vocab_size = len(self.vocab) + 1

        # Encoder
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
        input_encoder_m.add(Dropout(0.3))

        input_encoder_c = Sequential()
        input_encoder_c.add(
            Embedding(input_dim=vocab_size, output_dim=self.max_question_length)
        )
        input_encoder_c.add(Dropout(0.3))

        question_encoder = Sequential()
        question_encoder.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=64,
                input_length=self.max_question_length,
            )
        )
        question_encoder.add(Dropout(0.3))

        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation("softmax")(match)

        response = add([match, input_encoded_c])
        response = Permute((2, 1))(response)

        answer = concatenate([response, question_encoded])
        answer = LSTM(32)(answer)
        answer = Dropout(0.5)
        answer = Dense(vocab_size)(answer)
        answer = Activation("softmax")(answer)

        # Building Model
        model = Model([input_sequence, question], answer)
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # model.summary()
        history = model.fit(
            [inputs_train, queries_train],
            answer_train,
            batch_size=32,
            epochs=25,
            validation_data=([inputs_test, queries_test], answer_test),
        )

        self.plot_result(history=history)

        pred_results = model.predict(([inputs_test, queries_test]))
        val_max = np.argmax(pred_results[0])
        for key, value in self.tokenizer.word_index.items():
            if value == val_max:
                k = key # answer: no


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
